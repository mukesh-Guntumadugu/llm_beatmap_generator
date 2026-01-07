# LLM Beatmap Generator

A tool to generate StepMania (`.sm`) beatmaps from audio files using Large Language Models (LLM).

## Overview

This project uses audio analysis (via `librosa`) and LLMs (via Google Gemini) to create rhythm game charts.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mukesh-Guntumadugu/llm_beatmap_generator.git
    cd llm_beatmap_generator
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    Create a `.env` file and add your API key:
    ```
    GOOGLE_API_KEY=your_api_key_here
    ```

## Usage

```bash
python src/main.py --audio path/to/song.mp3 --difficulty Hard
```

## Structure

*   `src/`: Source code
    *   `src/main.py`: CLI entry point
    *   `src/audio_processor.py`: Audio analysis
    *   `src/generator.py`: LLM interaction
    *   `src/sm_writer.py`: File writer
