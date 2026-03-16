"""
mumu_interface.py
=================
Loads MuMu-LLaMA (M2UGen) and performs onset detection inference.

Requirements (on HPC cluster):
  - MuMu-LLaMA repo cloned at: <project_root>/MuMu-LLaMA/
  - LLaMA-2 7B weights at:     MuMu-LLaMA/ckpts/LLaMA/7B/
  - M2UGen checkpoint at:      MuMu-LLaMA/ckpts/MuMu-LLaMA-MusicGen/checkpoint.pth
  - conda activate mumullama (with MuMu-LLaMA requirements installed)
  - Run on GPU node via SLURM (A6000 recommended, needs ~48GB VRAM)

Repo structure note:
  The MuMu-LLaMA GitHub repo has an inner subdirectory also called MuMu-LLaMA/.
  The Python source (llama/, inference.py, etc.) lives in:
    <repo_root>/MuMu-LLaMA/MuMu-LLaMA/
  This is what we add to sys.path.
"""

import os
import sys
import argparse
import torch
import torchaudio

# ── Path Setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUMU_REPO_DIR = os.path.join(PROJECT_ROOT, "MuMu-LLaMA")          # repo root
MUMU_SRC_DIR  = os.path.join(MUMU_REPO_DIR, "MuMu-LLaMA")         # inner src dir
LLAMA_DIR     = os.path.join(MUMU_REPO_DIR, "ckpts", "LLaMA")     # contains 7B/
CHECKPOINT    = os.path.join(MUMU_REPO_DIR, "ckpts", "MuMu-LLaMA-MusicGen", "checkpoint.pth")

_model = None
_loaded = False

# Target audio sample rate expected by MERT (used inside MuMu-LLaMA)
MERT_SAMPLE_RATE = 24000


def _make_model_args() -> argparse.Namespace:
    """Build the argparse Namespace that MuMu_LLaMA.__init__ expects."""
    args = argparse.Namespace(
        llama_type="7B",
        llama_path=LLAMA_DIR,
        mert_path="m-a-p/MERT-v1-330M",
        vit_path="google/vit-base-patch16-224-in21k",
        vivit_path="google/vivit-b-16x2-kinetics400",
        music_decoder="musicgen",
        music_decoder_path="facebook/musicgen-medium",
        knn_dir=os.path.join(MUMU_REPO_DIR, "ckpts"),
        max_words=2048,
        num_gen_audio_tokens=8,
    )
    return args


def setup_mumu():
    """Load the MuMu-LLaMA model from disk."""
    global _model, _loaded

    # ── GPU check ───────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. MuMu-LLaMA requires a GPU (A6000 recommended).\n"
            "Submit via SLURM:  sbatch run_mumu.sh"
        )

    # ── Directory / file checks ──────────────────────────────────────────────
    if not os.path.isdir(MUMU_REPO_DIR):
        raise FileNotFoundError(
            f"MuMu-LLaMA repo not found at:\n  {MUMU_REPO_DIR}\n"
            "Clone it with: git clone https://github.com/shansongliu/MuMu-LLaMA.git"
        )
    if not os.path.isdir(MUMU_SRC_DIR):
        raise FileNotFoundError(
            f"MuMu-LLaMA inner source dir not found at:\n  {MUMU_SRC_DIR}\n"
            "Expected repo structure: MuMu-LLaMA/MuMu-LLaMA/llama/mumu_llama.py"
        )
    if not os.path.isdir(LLAMA_DIR):
        raise FileNotFoundError(
            f"LLaMA-2 weights not found at:\n  {LLAMA_DIR}\n"
            "Download them from HuggingFace: meta-llama/Llama-2-7b\n"
            "Expected structure: ckpts/LLaMA/7B/"
        )
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"MuMu-LLaMA checkpoint not found at:\n  {CHECKPOINT}\n"
            "Download from HuggingFace: M2UGen/M2UGen-MusicGen-medium"
        )

    # ── Add MuMu-LLaMA *inner* src dir to path so `from llama.mumu_llama import ...` works
    if MUMU_SRC_DIR not in sys.path:
        sys.path.insert(0, MUMU_SRC_DIR)

    print(f"Loading MuMu-LLaMA checkpoint: {CHECKPOINT}")
    print(f"LLaMA directory:               {LLAMA_DIR}")
    print(f"Source directory on path:      {MUMU_SRC_DIR}")

    try:
        from llama.mumu_llama import MuMu_LLaMA  # type: ignore

        llama_ckpt_dir      = os.path.join(LLAMA_DIR, "7B")
        llama_tokenizer_dir = llama_ckpt_dir      # tokenizer.model is inside 7B/ on this cluster
        model_args          = _make_model_args()

        _model = MuMu_LLaMA(
            llama_ckpt_dir=llama_ckpt_dir,
            llama_tokenizer=llama_tokenizer_dir,
            model_args=model_args,
            knn=False,
            stage=3,
            load_llama=False,
        )

        print("Loading checkpoint weights...")
        raw_ckpt = torch.load(CHECKPOINT, map_location="cpu")

        # Strip "module." prefix added by DDP training
        new_ckpt = {
            k.replace("module.", ""): v
            for k, v in raw_ckpt["model"].items()
        }
        load_result = _model.load_state_dict(new_ckpt, strict=False)
        if load_result.unexpected_keys:
            print(f"  ⚠️  Unexpected keys: {load_result.unexpected_keys[:5]} ...")

        _model = _model.cuda()
        _model.eval()
        _loaded = True
        print("✅ MuMu-LLaMA loaded successfully.")

    except ImportError as e:
        raise ImportError(
            f"Could not import MuMu-LLaMA model class: {e}\n"
            f"Make sure {MUMU_SRC_DIR} exists and contains llama/mumu_llama.py"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load MuMu-LLaMA: {e}")


def _load_audio_tensor(audio_path: str) -> torch.Tensor:
    """
    Load an audio file and resample to MERT_SAMPLE_RATE (24 kHz).
    Returns a 1-D float32 tensor on CPU.
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != MERT_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=MERT_SAMPLE_RATE)
        waveform = resampler(waveform)
    # Convert stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)   # shape: (num_samples,)


def generate_beatmap_with_mumu(audio_path: str, prompt: str) -> str:
    """
    Passes audio + prompt to MuMu-LLaMA and returns the text response.

    Parameters
    ----------
    audio_path : str
        Path to the audio file (.ogg / .mp3 / .wav)
    prompt : str
        Text prompt / instruction for the model

    Returns
    -------
    str
        The model's text response (should be a JSON onset array)
    """
    global _model, _loaded

    if not _loaded:
        setup_mumu()

    print(f"  Generating with MuMu-LLaMA for: {os.path.basename(audio_path)}")

    audio_tensor = _load_audio_tensor(audio_path)

    with torch.no_grad():
        # MuMu_LLaMA.generate() has a bug: when prompts[0] is a str it does
        #   self.tokenizer(x).input_ids[:, 1:]
        # which requires a 2D tensor but LlamaTokenizer returns a plain list.
        # Fix: pre-tokenize here and pass token ID lists so the str branch is skipped.
        token_ids = _model.tokenizer.encode(prompt)  # list[int], includes BOS
        token_ids = token_ids[1:]                    # drop BOS (matches [:, 1:])

        result = _model.generate(
            prompts=[token_ids],   # list of token-id lists, not strings
            audios=audio_tensor,
            max_gen_len=4096,
            temperature=0.1,
            top_p=0.75,
        )

    # result is [decoded_text] or [decoded_text, {'aud': [...]}]
    print(f"  [DEBUG] Raw MuMu-LLaMA output type: {type(result)}")
    print(f"  [DEBUG] Raw MuMu-LLaMA output content: {result}")
    
    return result[0] if isinstance(result, list) else str(result)
