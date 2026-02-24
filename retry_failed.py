"""
Retry script for songs that failed during the previous batch run.
Run this after batch_process_fraxtil.py to process only the failed songs.
"""
import os
import time
import csv
import datetime
from dotenv import load_dotenv
import librosa

from src.gemini import setup_gemini, generate_beatmap_csv, create_beatmap_prompt_cache

DIFFICULTY = "Hard"
MODEL_NAME = "gemini-pro-latest"

# ── Songs that failed in the previous run ────────────────────────────────────
BASE = "/Users/mukeshguntumadugu/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements"

FAILED_SONGS = [
    os.path.join(BASE, "Pain Game/Pain Game.ogg"),
    os.path.join(BASE, "Inner Universe (Extended Mix)/Inner Universe (Extended Mix).ogg"),
    os.path.join(BASE, "The Seconds/The Seconds.ogg"),
    os.path.join(BASE, "Call Me Maybe (Remix)/Call Me Maybe (Remix).ogg"),
    os.path.join(BASE, "Dead Hollywood/Dead Hollywood.ogg"),
    os.path.join(BASE, "Star Gear/Star Gear.ogg"),
    os.path.join(BASE, "Human Error/Human Error.ogg"),
    os.path.join(BASE, "Deal With It/Deal With It.ogg"),
    os.path.join(BASE, "No Beginning/No Beginning.ogg"),
]


def process_song(audio_path, cached_content_name=None):
    print(f"Processing: {os.path.basename(audio_path)}")
    try:
        duration = librosa.get_duration(path=audio_path)
        print(f"  Duration: {duration:.1f}s")

        cache_status = "cached prompt" if cached_content_name else "full prompt"
        print(f"  Sending to Gemini ({DIFFICULTY}, {MODEL_NAME}, {cache_status})...")
        rows = generate_beatmap_csv(
            audio_path=audio_path,
            duration=duration,
            difficulty=DIFFICULTY,
            model_name=MODEL_NAME,
            cached_content_name=cached_content_name
        )

        if not rows:
            print("   No data generated.")
            return

        dirname = os.path.dirname(audio_path)
        name_no_ext = os.path.splitext(os.path.basename(audio_path))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{name_no_ext}_{DIFFICULTY}_{MODEL_NAME}_{timestamp}"

        # Plain beatmap .txt
        beatmap_path = os.path.join(dirname, f"{base}.txt")
        with open(beatmap_path, 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(f"{row.notes}\n")
        print(f"  Beatmap  → {os.path.basename(beatmap_path)}")

        # Rich metadata .csv
        csv_path = os.path.join(dirname, f"{base}.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["time_ms", "beat_position", "notes",
                             "placement_type", "note_type", "confidence", "instrument"])
            for row in rows:
                writer.writerow([
                    row.time_ms,
                    f"{row.beat_position:.3f}",
                    row.notes,
                    row.placement_type,
                    row.note_type,
                    f"{row.confidence:.3f}",
                    row.instrument
                ])
        print(f"  Full CSV → {os.path.basename(csv_path)}")
        print("   Done.")

    except Exception as e:
        print(f"   Error: {e}")


def main():
    load_dotenv(override=True)
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        return

    setup_gemini(api_key)

    # Try to create a fresh cache for this retry run
    cache_name = create_beatmap_prompt_cache(
        difficulty=DIFFICULTY,
        model_name=MODEL_NAME,
        ttl_seconds=3600
    )
    if cache_name:
        print(f"Using context cache: {cache_name}\n")
    else:
        print("Running without cache (full prompt per song).\n")

    total = len(FAILED_SONGS)
    for i, audio_file in enumerate(FAILED_SONGS):
        if not os.path.exists(audio_file):
            print(f"[{i+1}/{total}] File not found, skipping: {audio_file}")
            continue
        print(f"\n[{i+1}/{total}] Retrying...")
        process_song(audio_file, cached_content_name=cache_name)
        if i < total - 1:
            time.sleep(2)

    print("\nRetry run complete.")


if __name__ == "__main__":
    main()
