import os
import sys
import torch
import warnings
warnings.filterwarnings("ignore")

# --- Bypass DeepSpeed NVCC Bug ---
from unittest.mock import MagicMock
try:
    import triton
except ImportError:
    sys.modules['triton'] = MagicMock()
sys.modules['triton.ops'] = MagicMock()
sys.modules['triton.ops.matmul_perf_model'] = MagicMock()
# ---------------------------------

DEEPRESONANCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepResonance", "code")
CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepResonance", "ckpt")
sys.path.append(DEEPRESONANCE_DIR)
os.chdir(DEEPRESONANCE_DIR)

from inference_deepresonance import DeepResonancePredict

def main():
    print("==================================================")
    print("      DeepResonance Interactive CLI Terminal      ")
    print("==================================================")
    print("Loading 12GB AI Model into VRAM... Please wait...")

    try:
        model_args = {
            'stage': 2,
            'mode': 'test',
            'project_path': DEEPRESONANCE_DIR,
            'dataset': 'musiccaps',
            'llm_path': os.path.join(CKPT_PATH, 'pretrained_ckpt', 'vicuna_ckpt', '7b_v0'),
            'imagebind_path': os.path.join(CKPT_PATH, 'pretrained_ckpt', 'imagebind_ckpt', 'huge'),
            'imagebind_version': 'huge',
            'max_length': 512,
            'max_output_length': 512,
            'num_clip_tokens': 77,
            'gen_emb_dim': 768,
            'preencoding_dropout': 0.1,
            'num_preencoding_layers': 1,
            'lora_r': 32,
            'lora_alpha': 32,
            'lora_dropout': 0.1,
            'freeze_lm': False,
            'freeze_input_proj': False,
            'freeze_output_proj': False,
            'prompt': '',
            'prellmfusion': True,
            'prellmfusion_dropout': 0.1,
            'num_prellmfusion_layers': 1,
            'imagebind_embs_seq': True,
            'topp': 1.0,
            'temp': 0.1,
            'ckpt_path': os.path.join(CKPT_PATH, 'DeepResonance_data_models', 'ckpt',
                                      'deepresonance_beta_delta_ckpt', 'delta_ckpt',
                                      'deepresonance', '7b_tiva_v0'),
        }
        dr_predictor = DeepResonancePredict(model_args)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nFailed to load model: {e}")
        return

    print("\n✅ DeepResonance Online! Type /exit to quit.\n")

    while True:
        try:
            print("─" * 50)
            audio_file = input("\n🎵 Audio file path (or Enter to skip): ").strip()

            if audio_file.lower() == "/exit":
                break

            prompt = input("🗣️  Your prompt: ").strip()
            if prompt.lower() == "/exit":
                break

            if not prompt:
                print("Cannot send empty prompt.")
                continue

            mm_names = [["audio"]]
            mm_root_path = "/tmp"

            if audio_file:
                if not os.path.exists(audio_file):
                    print("❌ Audio file not found!")
                    continue
                mm_paths = [[os.path.basename(audio_file)]]
                mm_root_path = os.path.dirname(os.path.abspath(audio_file))
                if "<Audio>" not in prompt:
                    prompt = f"<Audio>\n{prompt}"
            else:
                # DeepResonance REQUIRES audio — generate a silent dummy so text chat works
                import numpy as np
                import soundfile as sf
                dummy_path = "/tmp/DR_silent_dummy.wav"
                sf.write(dummy_path, np.zeros(22050, dtype=np.float32), 22050)
                mm_paths = [["DR_silent_dummy.wav"]]
                if "<Audio>" not in prompt:
                    prompt = f"<Audio>\n{prompt}"

            inputs = {
                "inputs": [prompt],
                "instructions": [prompt],
                "mm_names": mm_names,
                "mm_paths": mm_paths,
                "mm_root_path": mm_root_path,
                "outputs": [""],
            }

            print("\n🤖 Thinking...")
            response_text = dr_predictor.predict(
                inputs,
                max_tgt_len=1024,
                top_p=1.0,
                temperature=0.1,
                stops_id=[[835]],
            )

            if isinstance(response_text, list):
                response_text = response_text[0].split("\n###")[0]

            print(f"\n💬 RESPONSE:\n{response_text}\n")

            import gc
            gc.collect()
            torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
