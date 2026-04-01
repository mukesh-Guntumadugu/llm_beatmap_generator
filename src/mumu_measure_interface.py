import os
import sys
import argparse
import torch
import torchaudio
import torch.nn.functional as F

# Paths aligned with your cluster structure
MUMU_ROOT = "/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/MuMu-LLaMA"
LLAMA_DIR = "/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts/LLaMA"
MUMU_CKPT = "/data/mg546924/models/mumu-llama-lora-onsets/checkpoint_epoch4.pth" # Fallback to a fine-tuned if exists, or base if missing

# Add MuMu to system path so `llama.mumu_llama` works
sys.path.insert(0, MUMU_ROOT)

_mumu_model = None
_mumu_tokenizer = None

def initialize_mumu_model():
    """Load MuMu-LLaMA model and tokenizer globally into VRAM."""
    global _mumu_model, _mumu_tokenizer
    
    if _mumu_model is not None:
        return _mumu_model, _mumu_tokenizer
        
    print("Loading LLaMA tokenizer...", flush=True)
    try:
        from llama.mumu_llama import MuMu_LLaMA
        import llama
        from transformers import LlamaTokenizer
    except ImportError as e:
        print(f"Error importing MuMu modules. Are you in the deepresonance or qwenenv conda env? e: {e}")
        raise e

    _mumu_tokenizer = LlamaTokenizer.from_pretrained(LLAMA_DIR)

    model_args = argparse.Namespace(
        mert_path="m-a-p/MERT-v1-330M",
        vit_path="google/vit-base-patch16-224",
        vivit_path="google/vivit-b-16x2-kinetics400",
        music_decoder="musicgen",
        music_decoder_path="facebook/musicgen-small",
        max_words=512,
    )

    print("Loading MuMu-LLaMA Base Architecture...", flush=True)
    _mumu_model = MuMu_LLaMA(
        llama_ckpt_dir=os.path.join(LLAMA_DIR, "7B"),
        llama_tokenizer=LLAMA_DIR,
        model_args=model_args,
        knn_dir="/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts",
        stage=3,
    )

    # First attempt to load your LoRA checkpoint if it exists, otherwise use base MuMu
    target_ckpt = MUMU_CKPT
    if not os.path.exists(target_ckpt):
        target_ckpt = "/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts/MuMu-LLaMA-MusicGen/checkpoint.pth"
        print(f"⚠️  LoRA Checkpoint not found. Falling back to base: {target_ckpt}")

    print(f"Loading checkpoint weights: {target_ckpt}", flush=True)
    ckpt = torch.load(target_ckpt, map_location="cpu")
    
    # Check if this is a raw state dict or nested
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
        
    _mumu_model.load_state_dict(state_dict, strict=False)
    _mumu_model = _mumu_model.cuda().float()
    _mumu_model.eval()
    
    print("✅ MuMu-LLaMA fully loaded onto GPU.", flush=True)
    return _mumu_model, _mumu_tokenizer

def get_mumu_16_step_probabilities(audio_path, prompt, candidates, 
                                     temperature=1.0, 
                                     top_p=1.0, 
                                     min_p=0.0, 
                                     top_k=None,
                                     repetition_penalty=1.0,
                                     recent_history=None):
    """
    Evaluates the math forward pass of MuMu-LLaMA across exactly 16 possible step patterns.
    """
    global _mumu_model, _mumu_tokenizer
    if _mumu_model is None:
        initialize_mumu_model()
        
    import llama
        
    # ── 1. Process Audio into 24kHz Tensor ──
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 24000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=24000)
        audio_tensor = torch.mean(waveform, 0).unsqueeze(0).cuda()  # [1, seq]
    except Exception as e:
        print(f"Error loading audio {audio_path}: {e}")
        return {cand: 1.0/len(candidates) for cand in candidates}

    # Format the prompt using MuMu's native prompt wrapper
    input1_str = llama.utils.format_prompt(prompt)
    input1_ids = _mumu_tokenizer(input1_str).input_ids

    # ── 2. Calculate Logits / Negative Loss for all Candidates ──
    cand_scores = []
    
    for cand in candidates:
        input2_str = input1_str + cand
        input2_ids = _mumu_tokenizer(input2_str).input_ids
        
        examples_tensor = torch.tensor([input2_ids], dtype=torch.int64).cuda()
        labels_tensor = examples_tensor.clone()
        
        # In MuMu-LLaMA training, they masked the prompt portion with 0. 
        # (Though huggingface normally uses -100, MuMu's custom loss wrapper uses 0 or looks for padded 0s. 
        # Actually, let's use 0 because the train script uses: `labels[:len(input1_ids)] = 0`)
        labels_tensor[0, :len(input1_ids)] = 0

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # MuMu returns language loss (c_loss) and music tracking loss (m_loss)
                c_loss, m_loss = _mumu_model(examples_tensor, labels_tensor, audios=audio_tensor, music_caption=None)
            
            # The cross entropy loss returned is an average over the active labels.
            num_cand_tokens = len(input2_ids) - len(input1_ids)
            # Reconstruct sum of log probabilities
            seq_log_prob = -c_loss.item() * num_cand_tokens
            cand_scores.append(seq_log_prob)

    scores_tensor = torch.tensor(cand_scores, dtype=torch.float32)

    # ── 3. Dynamic Repetition Penalty ──
    if recent_history and repetition_penalty > 1.0:
        for i, cand in enumerate(candidates):
            if cand == "0000" and recent_history.count("0000") < 4:
                continue  # Free passes for rests
                
            count = recent_history.count(cand)
            if count > 0:
                dyn_penalty = 1.0 + (count * (repetition_penalty - 1.0))
                if cand == recent_history[-1]:
                    dyn_penalty *= 1.1
                scores_tensor[i] = scores_tensor[i] * dyn_penalty

    # ── 4. Temperature Scaling ──
    if temperature != 1.0 and temperature > 0.0:
        scores_tensor = scores_tensor / temperature

    # ── 5. Convert to Probabilities (Softmax) ──
    probs = F.softmax(scores_tensor, dim=-1)

    # ── 6. Advanced Filters (Min-P, Top-P, Top-K) ──
    if min_p > 0.0:
        mask = probs < min_p
        probs[mask] = 0.0

    if top_k is not None and top_k > 0 and top_k < len(candidates):
        topk_vals, topk_indices = torch.topk(probs, top_k)
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask[topk_indices] = False
        probs[mask] = 0.0

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0

    # Renormalize to ensure they sum to 1.0 after filtering
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = torch.ones_like(probs) / len(candidates)

    probs_dict = {candidates[i]: probs[i].item() for i in range(len(candidates))}
    
    # Sort dict highest to lowest chance
    probs_dict = dict(sorted(probs_dict.items(), key=lambda item: item[1], reverse=True))

    return probs_dict
