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

Install `ffmpeg` (required by Whisper):

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y ffmpeg

# macOS (Homebrew)
brew install ffmpeg
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

By default, grammar uses `--grammar-mode auto` (tries public API first to avoid heavy local download, then falls back to local if needed).

`--dataset-dir` can point to either:
- the `LJSpeech-1.1` directory itself, or
- its parent directory (the script will auto-detect `LJSpeech-1.1/wavs`).

> **Important:** `/path/to/...` in examples is a placeholder. Replace it with your real dataset path.

Quick examples:

```bash
# If your dataset is at /home/you/datasets/LJSpeech-1.1
python stt_grammar_pipeline.py --dataset-dir /home/you/datasets/LJSpeech-1.1

#if you want both plot and results csv file
python stt_grammar_pipeline.py --dataset-dir /home/you/datasets --grammar-mode local --output-csv sample_results.csv --plot

# If your dataset folder contains LJSpeech-1.1/
python stt_grammar_pipeline.py --dataset-dir /home/you/datasets
```


## Common command mistakes

If you see `bash: --output-csv: command not found`, it usually means a multi-line command was pasted without line-continuation backslashes.

Use either a **single-line** command:

```bash
python stt_grammar_pipeline.py --dataset-dir /home/you/datasets/LJSpeech-1.1 --output-csv results.csv --model base --language en --plot
```


Or keep backslashes at the end of each continued line:

```bash
python stt_grammar_pipeline.py \
  --dataset-dir /home/you/datasets/LJSpeech-1.1 \
  --output-csv results.csv \
  --model base \
  --language en \
  --plot
```



### Version / argument mismatch troubleshooting

If you run `--grammar-mode public` and see `unrecognized arguments: --grammar-mode public`, you are likely running an older copy of the script from a different folder.

Check which script/version you are executing:

```bash
python stt_grammar_pipeline.py --version
python stt_grammar_pipeline.py --help
```

`--help` should list `--grammar-mode {auto,local,public}`. If it does not, run from this repository folder and update your local files before retrying.

### Useful flags

- `--limit N` – process only first `N` files
- `--device cpu|cuda` – select Whisper device
- `--no-grammar` – skip grammar correction
- `--grammar-mode auto|public|local` – choose grammar backend (`public` avoids local 255MB download)
- `--plot` – generate `wer_comparison.png` and `corrections_per_file.png`

---


If your process gets terminated while initializing grammar correction, run either:

```bash
python stt_grammar_pipeline.py --dataset-dir /home/you/datasets/LJSpeech-1.1 --grammar-mode public
```

or (ASR-only):

```bash
python stt_grammar_pipeline.py --dataset-dir /home/you/datasets/LJSpeech-1.1 --no-grammar
```

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

## Screenshots Demo

<img width="1919" height="1199" alt="Screenshot 2026-02-24 191308" src="https://github.com/user-attachments/assets/ae55f71d-dc6b-4afe-b856-1e8ee616e806" />

<img width="1918" height="1196" alt="Screenshot 2026-02-24 190221" src="https://github.com/user-attachments/assets/45f1c5c2-cd64-43ec-a59f-4be387c7ca90" />

## Video Demo

without plot

https://github.com/user-attachments/assets/d8e27f0c-2fa0-4278-bf17-111032f19af9

With plot

https://github.com/user-attachments/assets/b9532ba2-f772-4bcc-b349-00ea4151f671

With result and plot

https://github.com/user-attachments/assets/460ebd68-6b4d-41cd-b729-322283d69418






