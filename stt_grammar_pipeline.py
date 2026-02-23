#!/usr/bin/env python3
"""Speech-to-Text + Grammar Correction pipeline for LJ Speech."""

from __future__ import annotations

import argparse
import difflib
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from jiwer import wer
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LJ Speech audio to text and apply grammar correction."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to LJSpeech-1.1")
    parser.add_argument("--output-csv", type=Path, default=Path("results.csv"), help="Output CSV path")
    parser.add_argument("--model", type=str, default="base", help="Whisper model name")
    parser.add_argument("--language", type=str, default="en", help="Language code for ASR")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files")
    parser.add_argument("--no-grammar", action="store_true", help="Disable grammar correction")
    parser.add_argument(
        "--grammar-mode",
        type=str,
        choices=["auto", "local", "public"],
        default="auto",
        help=(
            "Grammar backend: auto (try public then local), "
            "public (no local model download), or local"
        ),
    )
    parser.add_argument("--plot", action="store_true", help="Generate optional plots")
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> Dict[str, str]:
    """Load LJ Speech metadata.csv into {file_id: transcript}."""
    if not metadata_path.exists():
        return {}

    id_to_text: Dict[str, str] = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) >= 2:
                # metadata format: ID|transcript|normalized_transcript
                id_to_text[parts[0]] = parts[1].strip()
    return id_to_text


def init_asr_model(model_name: str, device: str):
    import whisper

    return whisper.load_model(model_name, device=device)


def init_grammar_tool(language: str, mode: str = "auto"):
    import language_tool_python

    if mode not in {"auto", "local", "public"}:
        raise ValueError(f"Unsupported grammar mode: {mode}")

    if mode in {"auto", "public"}:
        try:
            return language_tool_python.LanguageToolPublicAPI(language)
        except Exception as exc:
            if mode == "public":
                raise RuntimeError(
                    "Failed to initialize LanguageTool Public API. "
                    "Use --grammar-mode local or --no-grammar."
                ) from exc
            warnings.warn(
                f"Public API unavailable ({exc}); falling back to local LanguageTool.",
                RuntimeWarning,
            )

    return language_tool_python.LanguageTool(language)


def transcribe_audio(model, audio_path: Path, language: str) -> str:
    result = model.transcribe(str(audio_path), language=language, fp16=False)
    return result.get("text", "").strip()


def grammar_correct(tool, text: str) -> str:
    if not text:
        return text
    return tool.correct(text)


def highlight_corrections(original: str, corrected: str) -> str:
    """Mark inserted/replaced words in corrected text with [*word*]."""
    o_tokens = original.split()
    c_tokens = corrected.split()

    matcher = difflib.SequenceMatcher(a=o_tokens, b=c_tokens)
    highlighted: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            highlighted.extend(c_tokens[j1:j2])
        else:
            highlighted.extend([f"[*{tok}*]" for tok in c_tokens[j1:j2]])

    return " ".join(highlighted)


def count_corrections(original: str, corrected: str) -> int:
    o_tokens = original.split()
    c_tokens = corrected.split()
    matcher = difflib.SequenceMatcher(a=o_tokens, b=c_tokens)

    count = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            count += max(i2 - i1, j2 - j1)
    return count


def compute_wer_pair(reference: Optional[str], hypothesis: str) -> Optional[float]:
    if not reference:
        return None
    return float(wer(reference, hypothesis))


def maybe_plot(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    plot_df = df.copy()

    if "wer_raw" in plot_df.columns and plot_df["wer_raw"].notna().any():
        idx = range(len(plot_df))
        width = 0.4
        plt.figure(figsize=(12, 5))
        plt.bar([i - width / 2 for i in idx], plot_df["wer_raw"], width=width, label="WER Raw")
        plt.bar(
            [i + width / 2 for i in idx],
            plot_df["wer_corrected"],
            width=width,
            label="WER Corrected",
        )
        plt.xlabel("File index")
        plt.ylabel("WER")
        plt.title("WER Comparison (Raw ASR vs Grammar Corrected)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("wer_comparison.png", dpi=150)
        plt.close()

    plt.figure(figsize=(12, 5))
    plt.bar(range(len(plot_df)), plot_df["num_corrections"])
    plt.xlabel("File index")
    plt.ylabel("Corrections")
    plt.title("Number of Corrections per File")
    plt.tight_layout()
    plt.savefig("corrections_per_file.png", dpi=150)
    plt.close()




def resolve_dataset_paths(dataset_dir: Path) -> tuple[Path, Path]:
    """Resolve LJ Speech directory whether user points to root or LJSpeech-1.1."""
    raw_arg = str(dataset_dir)
    if raw_arg.startswith("/path/to/") or raw_arg == "/path/to/LJSpeech-1.1":
        raise FileNotFoundError(
            "You passed a placeholder path ('/path/to/...'). "
            "Replace it with the real location of your dataset, e.g. "
            "--dataset-dir /home/you/datasets/LJSpeech-1.1"
        )

    dataset_dir = dataset_dir.expanduser().resolve()

    direct_wavs = dataset_dir / "wavs"
    direct_meta = dataset_dir / "metadata.csv"
    if direct_wavs.exists():
        return direct_wavs, direct_meta

    nested_dir = dataset_dir / "LJSpeech-1.1"
    nested_wavs = nested_dir / "wavs"
    nested_meta = nested_dir / "metadata.csv"
    if nested_wavs.exists():
        return nested_wavs, nested_meta

    raise FileNotFoundError(
        "Could not find LJ Speech wav files. "
        f"Checked: '{direct_wavs}' and '{nested_wavs}'. "
        "Pass --dataset-dir as either the LJSpeech-1.1 folder itself "
        "or its parent directory. Also ensure you are not using the literal "
        "example placeholder '/path/to/...'."
    )

def main() -> None:
    args = parse_args()

    wav_dir, metadata_path = resolve_dataset_paths(args.dataset_dir)

    id_to_gt = load_metadata(metadata_path)

    wav_files = sorted(wav_dir.glob("*.wav"))
    if args.limit is not None:
        wav_files = wav_files[: args.limit]

    if not wav_files:
        raise RuntimeError("No WAV files found to process.")

    print(f"Loading Whisper model '{args.model}' on {args.device}...")
    asr_model = init_asr_model(args.model, args.device)

    grammar_tool = None
    if not args.no_grammar:
        print("Initializing grammar correction tool...")
        grammar_tool = init_grammar_tool(args.language, args.grammar_mode)

    rows = []

    for wav_file in tqdm(wav_files, desc="Processing audio"):
        file_id = wav_file.stem
        gt = id_to_gt.get(file_id)

        raw_text = transcribe_audio(asr_model, wav_file, language=args.language)
        corrected_text = (
            grammar_correct(grammar_tool, raw_text) if grammar_tool is not None else raw_text
        )

        highlighted = highlight_corrections(raw_text, corrected_text)
        n_corrections = count_corrections(raw_text, corrected_text)

        wer_raw = compute_wer_pair(gt, raw_text)
        wer_corrected = compute_wer_pair(gt, corrected_text)

        rows.append(
            {
                "file_id": file_id,
                "ground_truth": gt,
                "raw_asr_output": raw_text,
                "corrected_output": corrected_text,
                "highlighted_corrected_output": highlighted,
                "num_corrections": n_corrections,
                "wer_raw": wer_raw,
                "wer_corrected": wer_corrected,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)

    print("\n--- Sample Outputs ---")
    for _, r in df.head(5).iterrows():
        print(f"\nFile: {r['file_id']}")
        print(f"Original ASR : {r['raw_asr_output']}")
        print(f"Corrected    : {r['corrected_output']}")

    if df["wer_raw"].notna().any():
        mean_raw = df["wer_raw"].mean()
        mean_corr = df["wer_corrected"].mean()
        improvement = ((mean_raw - mean_corr) / mean_raw * 100.0) if mean_raw > 0 else 0.0

        print("\n--- Evaluation ---")
        print(f"Average WER (raw)      : {mean_raw:.4f}")
        print(f"Average WER (corrected): {mean_corr:.4f}")
        print(f"Improvement            : {improvement:.2f}%")
    else:
        print("\nGround truth not available; skipped WER computation.")

    if args.plot:
        maybe_plot(df)
        print("Saved plots: wer_comparison.png, corrections_per_file.png")

    print(f"\nSaved full results to: {args.output_csv}")


if __name__ == "__main__":
    main()
