"""
mumu_interface.py
=================
Loads MuMu-LLaMA (M2UGen) and performs onset detection inference.

Requirements (on HPC cluster):
  - MuMu-LLaMA repo cloned at: <project_root>/MuMu-LLaMA/
  - LLaMA-2 7B weights at:     MuMu-LLaMA/ckpts/LLaMA/7B/
  - M2UGen checkpoint at:      MuMu-LLaMA/ckpts/MuMu-LLaMA/checkpoint.pth
  - conda activate mumullama (with MuMu-LLaMA requirements installed)
  - Run on GPU node via SLURM (A6000 recommended, needs ~48GB VRAM)
"""

import os
import sys
import torch

# ── Path Setup ─────────────────────────────────────────────────────────────────
# The MuMu-LLaMA repo is cloned inside the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUMU_REPO_DIR = os.path.join(PROJECT_ROOT, "MuMu-LLaMA")
LLAMA_DIR     = os.path.join(MUMU_REPO_DIR, "ckpts", "LLaMA")
CHECKPOINT    = os.path.join(MUMU_REPO_DIR, "ckpts", "MuMu-LLaMA", "checkpoint.pth")

_model = None
_loaded = False


def setup_mumu():
    """Load the MuMu-LLaMA model from disk."""
    global _model, _loaded

    # ── GPU check ───────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected. MuMu-LLaMA requires a GPU (A6000 recommended).\n"
            "Submit via SLURM:  sbatch run_mumu.sh"
        )

    # ── Directory checks ─────────────────────────────────────────────────────
    if not os.path.isdir(MUMU_REPO_DIR):
        raise FileNotFoundError(
            f"MuMu-LLaMA repo not found at:\n  {MUMU_REPO_DIR}\n"
            "Clone it with: git clone https://github.com/shansongliu/MuMu-LLaMA.git"
        )
    if not os.path.isdir(LLAMA_DIR):
        raise FileNotFoundError(
            f"LLaMA-2 weights not found at:\n  {LLAMA_DIR}\n"
            "Download them from HuggingFace: meta-llama/Llama-2-7b"
        )
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"MuMu-LLaMA checkpoint not found at:\n  {CHECKPOINT}\n"
            "Download from HuggingFace: M2UGen/M2UGen-MusicGen-medium"
        )

    # ── Add MuMu-LLaMA source to path so we can import their modules ─────────
    if MUMU_REPO_DIR not in sys.path:
        sys.path.insert(0, MUMU_REPO_DIR)

    print(f"Loading MuMu-LLaMA checkpoint: {CHECKPOINT}")
    print(f"LLaMA directory:               {LLAMA_DIR}")

    try:
        # MuMu-LLaMA uses their own model class — import after path is set up
        from model import MuMuLLaMA  # type: ignore  (from MuMu-LLaMA repo)

        _model = MuMuLLaMA(
            llama_path=LLAMA_DIR,
            music_decoder="musicgen",
            music_decoder_path="facebook/musicgen-medium",
        )
        checkpoint = torch.load(CHECKPOINT, map_location="cpu")
        _model.load_state_dict(checkpoint["model"], strict=False)
        _model = _model.cuda()
        _model.eval()
        _loaded = True
        print("✅ MuMu-LLaMA loaded successfully.")

    except ImportError as e:
        raise ImportError(
            f"Could not import MuMu-LLaMA model class: {e}\n"
            f"Make sure you are running from inside the conda 'mumullama' environment\n"
            f"and that {MUMU_REPO_DIR} contains the MuMu-LLaMA source code."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load MuMu-LLaMA: {e}")


def generate_beatmap_with_mumu(audio_path: str, prompt: str) -> str:
    """
    Passes audio + prompt to MuMu-LLaMA and returns the text response.
    """
    global _model, _loaded

    if not _loaded:
        setup_mumu()

    print(f"  Generating with MuMu-LLaMA for: {os.path.basename(audio_path)}")

    try:
        # MuMu-LLaMA / MU-LLaMA uses a generate() method that accepts
        # audio_path and a text prompt directly
        with torch.no_grad():
            response = _model.generate(
                audio=audio_path,
                prompt=prompt,
                max_new_tokens=4096,
            )
        return response

    except AttributeError:
        # Older versions of the model use multimodal_generate
        try:
            from inference import multimodal_generate  # type: ignore
            response = multimodal_generate(
                model=_model,
                audio_path=audio_path,
                question=prompt,
                max_new_tokens=4096,
            )
            return response
        except Exception as e2:
            raise RuntimeError(f"MuMu-LLaMA inference failed: {e2}")

    except Exception as e:
        raise RuntimeError(f"MuMu-LLaMA generate() failed: {e}")
