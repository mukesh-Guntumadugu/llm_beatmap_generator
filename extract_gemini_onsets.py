"""
extract_gemini_onsets.py
========================
Sends each of the 20 Fraxtil songs to the Gemini model and asks it to
predict audio onset times in milliseconds. Saves results to CSV files.

Output filename: Gemini_onsets_<SongName>_<ddmmyyyyHHMMSS>.csv
Output columns : onset_index, onset_ms

Model choice:
  gemini-2.0-flash-001  — large context window, free-tier friendly (default)
  gemini-1.5-pro        — even larger context, may require paid plan
  gemini-pro-latest     — requires paid plan (free tier quota = 0)

Setup:
  Add your API key to the .env file at the project root:
      GOOGLE_API_KEY=your_key_here

Usage:
  python3 extract_gemini_onsets.py
  python3 extract_gemini_onsets.py --model gemini-2.0-flash-001
"""

import os
import re
import csv
import sys
import json
import time
import datetime
import argparse
import librosa
from typing import Optional, List
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

DEFAULT_MODEL = "gemini-3.1-pro-preview"  # confirmed available for this API key
# Alternatives: "gemini-2.0-flash-001" (1M context, faster/cheaper)

# ── Gemini client setup ───────────────────────────────────────────────────────

_client: Optional[genai.Client] = None

def is_hallucinated_loop(onsets: List[float]) -> bool:
    if not onsets or len(onsets) < 10:
        return False
    if len(onsets) > 150:
        return True 
    times = onsets
    deltas = [round(times[j+1] - times[j], 1) for j in range(len(times)-1)]
    for j in range(len(deltas) - 8):
        window = deltas[j:j+8]
        if all(w == window[0] for w in window):
            return True
    return False

def setup_gemini() -> genai.Client:
    """Initialise and return the Gemini client, reading the API key from env."""
    global _client
    if _client is not None:
        return _client
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not set.\n"
            "Add it to your .env file or export it in your shell."
        )
    _client = genai.Client(api_key=api_key)
    print(f"Gemini client initialised.")
    print(f"   API key: {api_key[:8]}...{api_key[-4:]}  (from .env)")
    return _client


# Static system instruction (cached once, reused for all 20 songs)
# Must be ≥ 1024 tokens for Gemini context caching to activate.
ONSET_SYSTEM_INSTRUCTION = (
    "You are an expert music analyst and audio engineer specializing in precise "
    "onset detection for rhythm game chart generation.\n\n"

    "## Your Task\n"
    "Listen to the audio carefully and extract the specific musical burst onsets.\n"
    "You must follow these strict rules to prevent hallucinations:\n"
    "1. Do not give me a bare list of numbers. Format each line exactly as: [Hit] : [Onset Timestamp in seconds]\n"
    "2. CRITICAL: Do not interpolate, guess, auto-fill, or count up sequentially. Only output the exact numbers found in the audio.\n"
    "3. Do not make up extra data if the audio is sparse. Stop generating when you reach the end."
)


def build_system_prompt(duration_sec: float) -> str:
    """Short per-song prompt sent alongside the audio. 
    Used when context cache is active (system instruction is already cached)."""
    return (
        f"The audio is {duration_sec:.1f} seconds long. "
        "Extract all musical onsets in '[Hit] : [Seconds]' format."
    )


def build_onset_prompt(duration_sec: float) -> str:
    """Full prompt used when context caching is NOT available."""
    return ONSET_SYSTEM_INSTRUCTION + "\n\n" + build_system_prompt(duration_sec)


def create_onset_cache(
    client: genai.Client,
    model_name: str,
    ttl_seconds: int = 3600
) -> Optional[str]:
    """
    Create a Gemini context cache for the static onset system instruction.
    Returns the cache name (string) to pass to generate_content, or None if
    caching fails (e.g., unsupported model, content too small, quota issue).

    The cache is valid for ttl_seconds (default 1 hour = entire batch run).
    Requires gemini-1.5-pro, gemini-1.5-flash, or gemini-2.0-flash-001.
    """
    MIN_CACHE_TOKENS = 1024  # Gemini's minimum token count for context caching

    try:
        # ── Step 1: Count tokens first ────────────────────────────────────────
        token_resp = client.models.count_tokens(
            model=model_name,
            contents=ONSET_SYSTEM_INSTRUCTION
        )
        token_count = token_resp.total_tokens
        print(f"  💾 Checking cache eligibility: {token_count} tokens ", end="", flush=True)

        if token_count < MIN_CACHE_TOKENS:
            print(f"(< {MIN_CACHE_TOKENS} minimum → skipping cache, using full prompt)")
            return None

        # ── Step 2: Token count meets threshold — create the cache ────────────
        print(f"(≥ {MIN_CACHE_TOKENS} → creating cache, TTL={ttl_seconds}s)...", end="", flush=True)
        cached = client.caches.create(
            model=model_name,
            config={
                "system_instruction": ONSET_SYSTEM_INSTRUCTION,
                "ttl": f"{ttl_seconds}s",
            }
        )
        print(f" ✅ Cache created: {cached.name}")
        return cached.name
    except Exception as e:
        print(f"\n  ⚠️  Context cache skipped: {e}")
        return None



# ── Helpers ───────────────────────────────────────────────────────────────────

def find_audio_file(song_dir: str) -> Optional[str]:
    """Return the first .ogg/.mp3/.wav found in a song directory."""
    for f in os.listdir(song_dir):
        if f.lower().endswith((".ogg", ".mp3", ".wav")):
            return os.path.join(song_dir, f)
    return None


def parse_onsets_from_response(response_text: str) -> List[float]:
    """
    Extract onset times in milliseconds from Gemini's response.

    Handles three output formats (tried in order):
      1. JSON array of numbers       (e.g. [0, 125, 250, 500])
      2. JSON objects with time_ms   (e.g. {"time_ms": 500.0, ...})
      3. Numbers in plain text       (fallback regex scan)
    """
    text = response_text.strip()
    # Strip markdown code fences
    text_clean = re.sub(r"```[\w]*\n?", "", text).strip()

    onsets: List[float] = []

    # 1. Anti-Hallucination Secure Anchored Parsing (Seconds)
    found_matches = re.findall(r':\s*([0-9]+\.?[0-9]*)', text_clean)
    if found_matches:
        for val in found_matches:
            try:
                ms = float(val) * 1000.0
                if 0.0 <= ms <= 600_000:
                    onsets.append(round(ms, 2))
            except ValueError:
                pass
        if onsets:
            return sorted(set(onsets))

    # ── Strategy 1: JSON array of numbers ────────────────────────────────────
    try:
        arr = json.loads(text_clean)
        if isinstance(arr, list):
            for val in arr:
                try:
                    ms = float(val)
                    if 0.0 <= ms <= 600_000:
                        onsets.append(round(ms, 2))
                except (ValueError, TypeError):
                    pass
            if onsets:
                return sorted(set(onsets))
    except (json.JSONDecodeError, ValueError):
        pass

    # ── Strategy 1b: Find embedded array with regex ───────────────────────────
    try:
        match = re.search(r'\[([\d.,\s]+)\]', text_clean)
        if match:
            arr = json.loads('[' + match.group(1) + ']')
            for val in arr:
                ms = float(val)
                if 0.0 <= ms <= 600_000:
                    onsets.append(round(ms, 2))
            if onsets:
                return sorted(set(onsets))
    except Exception:
        pass

    # ── Strategy 2: JSON objects with time_ms key ─────────────────────────────
    try:
        json_objects = re.findall(r'\{[^{}]+\}', text_clean)
        for obj_str in json_objects:
            try:
                obj = json.loads(obj_str)
                if 'time_ms' in obj:
                    ms = float(obj['time_ms'])
                    if 0.0 <= ms <= 600_000:
                        onsets.append(round(ms, 2))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        if onsets:
            return sorted(set(onsets))
    except Exception:
        pass

    # ── Strategy 3: Regex scan for bare numbers (fallback) ────────────────────
    numbers = re.findall(r'\b(\d+(?:[.,]\d+)?)\b', text_clean)
    for n in numbers:
        n = n.replace(',', '.')
        try:
            ms = float(n)
            if 0.0 <= ms <= 600_000:
                onsets.append(round(ms, 2))
        except ValueError:
            pass

    return sorted(set(onsets))


def save_onsets_csv(onset_ms: List[float], song_name: str, out_dir: str) -> str:
    """Save onsets to a CSV file and return the file path."""
    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    safe_name = song_name.replace(" ", "_").replace("/", "-")
    filename = f"Gemini_onsets_{safe_name}_{timestamp}.csv"
    filepath = os.path.join(out_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["onset_index", "onset_ms"])
        for idx, ms in enumerate(onset_ms):
            writer.writerow([idx, ms])

    return filepath


def query_gemini_for_onsets(
    audio_path: str,
    duration_sec: float,
    model_name: str,
    client: genai.Client,
    max_retries: int = 3,
    cache_name: Optional[str] = None
) -> str:
    """
    Upload audio to Gemini Files API, send the onset detection prompt,
    and return the raw response text.

    If cache_name is provided (from create_onset_cache), the static system
    instruction is served from cache — only the audio + short per-song
    duration hint is sent each call, saving tokens.

    Automatically retries on 429 RESOURCE_EXHAUSTED errors using the
    retryDelay hint from the API response.
    """
    import re as _re

    # Choose prompt based on whether cache is active
    if cache_name:
        prompt = build_system_prompt(duration_sec)   # short per-song prompt only
    else:
        prompt = build_onset_prompt(duration_sec)    # full prompt (no cache)

    for attempt in range(1, max_retries + 1):
        try:
            # Upload the audio file
            audio_file = client.files.upload(file=audio_path)

            # Wait for processing
            while audio_file.state == "PROCESSING":
                time.sleep(1)
                audio_file = client.files.get(name=audio_file.name)

            if audio_file.state != "ACTIVE":
                raise RuntimeError(f"File upload failed with state: {audio_file.state}")

            # Generate response
            if cache_name:
                config = types.GenerateContentConfig(
                    cached_content=cache_name,
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",        threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT",  threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",         threshold="BLOCK_NONE"),
                    ]
                )
            else:
                config = types.GenerateContentConfig(
                    safety_settings=[
                        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",        threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT",  threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT",  threshold="BLOCK_NONE"),
                        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",         threshold="BLOCK_NONE"),
                    ]
                )

            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, audio_file],
                config=config
            )

            # Clean up the uploaded file (best effort)
            try:
                client.files.delete(name=audio_file.name)
            except Exception:
                pass

            return response.text or ""

        except Exception as e:
            err_str = str(e)

            # ── Always print the full API error response ──────────────────────
            print(f"\n\n  {'='*60}")
            print(f"  FULL API ERROR RESPONSE (attempt {attempt}/{max_retries}):")
            print(f"  {'='*60}")
            print(err_str)
            print(f"  {'='*60}\n")

            # ── Handle 429 rate-limit / quota exhaustion ──────────────────────
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Try to extract the retryDelay seconds from the error message
                wait_sec = 60  # default fallback
                match = _re.search(r'retryDelay.*?"(\d+)s"', err_str)
                if match:
                    wait_sec = int(match.group(1)) + 5  # add 5s buffer
                else:
                    match = _re.search(r'retry in ([\d.]+)s', err_str)
                    if match:
                        wait_sec = int(float(match.group(1))) + 5

                if attempt < max_retries:
                    print(f"\n  ⏳ 429 rate limit — waiting {wait_sec}s before retry "
                          f"(attempt {attempt}/{max_retries})...", end="", flush=True)
                    time.sleep(wait_sec)
                    continue
                else:
                    raise RuntimeError(
                        f"429 quota exhausted after {max_retries} attempts.\n"
                        f"  Tip: consider using --model gemini-2.0-flash-001 (higher free-tier limits)\n"
                        f"  or upgrade to a paid Gemini API plan."
                    ) from e

            # Any other error — re-raise immediately
            raise

    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract audio onsets using Gemini for all Fraxtil songs."
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--delay", type=float, default=90.0,
        help="Seconds to wait between songs to avoid rate limits (default: 90)"
    )
    parser.add_argument(
        "--initial-wait", type=float, default=60.0,
        help="Seconds to wait before the FIRST request (lets quota window reset). Default: 60"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries on 429 errors per song (default: 3)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print Gemini's raw response for each song (useful for debugging)"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Disable context caching (send full prompt each time)"
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="1-based index of first song to process (default: 1)"
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="1-based index of last song to process (default: last song)"
    )
    args = parser.parse_args()

    # Initialise Gemini client
    try:
        client = setup_gemini()
    except RuntimeError as e:
        print(f"❌ {e}")
        return

    if not os.path.isdir(BASE_DIR):
        print(f"❌ Dataset directory not found:\n   {BASE_DIR}")
        return

    song_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
        and not d.startswith("_") and not d.startswith(".")
    ])

    # Apply --start / --end slice (1-based, inclusive)
    start_idx = max(0, args.start - 1)
    end_idx   = args.end if args.end is not None else len(song_dirs)
    song_dirs = song_dirs[start_idx:end_idx]

    print(f"\nModel  : {args.model}")
    print(f"Songs  : {len(song_dirs)} (songs {start_idx+1}–{start_idx+len(song_dirs)} of full list)")
    print(f"Delay  : {args.delay:.0f}s between songs | Initial wait: {args.initial_wait:.0f}s")
    print(f"Cache  : {'disabled (--no-cache)' if args.no_cache else 'enabled (system instruction)'} ")
    print(f"Output : Gemini_onsets_<SongName>_<timestamp>.csv\n")

    # Try to create context cache (once, before the song loop)
    cache_name: Optional[str] = None
    if not args.no_cache:
        cache_name = create_onset_cache(client, args.model, ttl_seconds=3600)

    # Initial wait — gives the per-minute quota window time to reset after test runs
    if args.initial_wait > 0:
        print(f"\u23f3 Waiting {args.initial_wait:.0f}s before first request "
              f"(clearing any leftover rate-limit window)...")
        elapsed = 0.0
        while elapsed < args.initial_wait:
            chunk = min(5.0, args.initial_wait - elapsed)
            time.sleep(chunk)
            elapsed += chunk
            remaining = args.initial_wait - elapsed
            if remaining > 0:
                print(f"   {remaining:.0f}s remaining...  ", end="\r", flush=True)
        print("✅ Ready — starting now." + " " * 20)
        print()

    print(f"{'Song':<45} {'# Onsets':>10}  {'Output file'}")
    print("─" * 110)

    total_songs = 0

    for i, song_name in enumerate(song_dirs):
        song_dir  = os.path.join(BASE_DIR, song_name)
        audio_path = find_audio_file(song_dir)

        if audio_path is None:
            print(f"  WARNING: [{i+1}/{len(song_dirs)}] No audio file in: {song_name}")
            continue

        print(f"  [{i+1}/{len(song_dirs)}] {song_name} ...", end="", flush=True)

        max_retries = 3
        onset_ms = []
        response_text = ""

        for attempt in range(max_retries):
            try:
                duration_sec = librosa.get_duration(path=audio_path)
                response_text = query_gemini_for_onsets(
                    audio_path, duration_sec, args.model, client,
                    max_retries=args.max_retries,
                    cache_name=cache_name
                )

                if args.verbose:
                    preview = response_text.replace("\n", " ")[:500]
                    print(f"\n  📨 Raw response ({len(response_text)} chars): {preview}")
                    print()

                if not response_text.strip():
                    print(f"\n  WARNING: Empty response for '{song_name}'")
                    continue

                onset_ms = parse_onsets_from_response(response_text)

                if onset_ms and is_hallucinated_loop(onset_ms):
                    print(f"\n  [!] HALLUCINATION CAUGHT: Neural counting drift detected! Retrying...")
                    onset_ms = []
                    continue

                if onset_ms:
                    break

            except Exception as e:
                print(f"\n  ERROR: {e}")
                if attempt == max_retries - 1:
                    break

        if not onset_ms:
            print(f"\n  WARNING: No parseable onsets for '{song_name}'")
            ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
            raw_path = os.path.join(
                song_dir,
                f"Gemini_onsets_RAW_{song_name.replace(' ', '_')}_{ts}.txt"
            )
            with open(raw_path, "w", encoding="utf-8") as f:
                f.write(response_text)
            print(f"     Raw saved -> {raw_path}")
            continue

        out_path = save_onsets_csv(onset_ms, song_name, song_dir)
        rel_out  = os.path.relpath(out_path, BASE_DIR)
        print(f"  -> {song_name:<43} {len(onset_ms):>10,}  {rel_out}")
        total_songs += 1

        # Respectful pause between API calls
        if i < len(song_dirs) - 1:   # no need to wait after the last song
            print(f"     ⏸  Waiting {args.delay:.0f}s before next song...", end="\r")
            time.sleep(args.delay)

    print("─" * 110)
    print(f"\n✅  Completed {total_songs}/{len(song_dirs)} songs.\n")


if __name__ == "__main__":
    main()
