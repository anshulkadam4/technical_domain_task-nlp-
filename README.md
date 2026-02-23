# NLP Task 1: Speech-to-Text with Grammar Correction

This repository contains a complete, runnable pipeline for:

1. Converting audio from the **LJ Speech dataset** into text using ASR (Whisper).
2. Applying **grammar correction** on the ASR output.
3. Displaying and exporting both:
   - Original ASR output
   - Grammar-corrected output
4. Evaluating quality with **Word Error Rate (WER)** (when ground truth is available in `metadata.csv`).
5. (Optional) Producing visualizations for error comparison and number of corrections.

---

## Features

- **ASR** via OpenAI Whisper (`tiny`, `base`, `small`, etc.)
- Batch processing of all `.wav` files in `LJSpeech-1.1/wavs/`
- Grammar correction via `language_tool_python`
- WER before/after grammar correction
- Improvement percentage reporting
- Per-file CSV report with highlighted corrections
- Optional plots:
  - WER before vs after
  - Number of corrections per file

---

## Project Files

- `stt_grammar_pipeline.py` – main pipeline script
- `requirements.txt` – Python dependencies

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset

Download LJ Speech from:

- https://keithito.com/LJ-Speech-Dataset/

Expected structure:

```text
LJSpeech-1.1/
├── metadata.csv
└── wavs/
    ├── LJ001-0001.wav
    ├── LJ001-0002.wav
    └── ...
```

---

## Usage

```bash
python stt_grammar_pipeline.py \
  --dataset-dir /absolute/path/to/LJSpeech-1.1 \
  --output-csv results.csv \
  --model base \
  --language en \
  --plot
```

`--dataset-dir` can point to either:
- the `LJSpeech-1.1` directory itself, or
- its parent directory (the script will auto-detect `LJSpeech-1.1/wavs`).

### Useful flags

- `--limit N` – process only first `N` files
- `--device cpu|cuda` – select Whisper device
- `--no-grammar` – skip grammar correction
- `--plot` – generate `wer_comparison.png` and `corrections_per_file.png`

---

## Output

The script prints summary metrics and saves a CSV containing:

- File name
- Ground truth transcript (if present)
- Original ASR output
- Grammar-corrected output
- Highlighted corrected text
- Per-file WER before and after correction
- Number of corrections

---

## Notes

- First run of Whisper and LanguageTool may download model assets.
- Grammar correction can improve readability, but WER may improve or worsen depending on how corrections align with ground truth.
