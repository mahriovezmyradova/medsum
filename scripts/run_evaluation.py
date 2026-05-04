#!/usr/bin/env python3
"""
Unified evaluation script – ASR × Summarizer comparison.

Outputs (data/outputs/evaluation/):
  asr_results.xlsx            – per-sample WER / CER / MTER per ASR model
  summarization_results.xlsx  – ROUGE / BLEU / medical-preservation per combo
  ai_impact.xlsx              – delta metrics: AI-enhanced vs raw
  best_duo.json               – best ASR+summarizer combination

Usage
-----
  python scripts/run_evaluation.py
  python scripts/run_evaluation.py --samples 30 --splits test
  python scripts/run_evaluation.py --skip-asr   # reuse cached ASR results
  python scripts/run_evaluation.py --no-ai       # skip the GPT step
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.medical.terminology import extract_medical_terms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATASET_PATH = Path(os.getenv("MULTIMED_PATH", "/Users/mahriovezmyradova/MultiMed_dataset/German"))
OUTPUT_DIR   = ROOT / "data" / "outputs" / "evaluation"
CACHE_DIR    = OUTPUT_DIR / "cache"

ASR_CONFIGS = [
    {"type": "whisper",  "name": "whisper_tiny",  "model_size": "tiny",  "language": "de"},
    {"type": "whisper",  "name": "whisper_base",  "model_size": "base",  "language": "de"},
    {"type": "whisper",  "name": "whisper_small", "model_size": "small", "language": "de"},
    {
        "type": "wav2vec2", "name": "wav2vec2_facebook",
        "model_name": "facebook/wav2vec2-large-xlsr-53-german", "language": "de",
    },
    {
        "type": "wav2vec2", "name": "wav2vec2_jonatasgrosman",
        "model_name": "jonatasgrosman/wav2vec2-large-xlsr-53-german", "language": "de",
    },
]

SUMMARIZER_CONFIGS = [
    {"type": "extractive", "name": "extractive_bert", "num_sentences": 3},
    {
        "type": "bart", "name": "bart_german",
        "model_name": "philschmid/bart-large-german-samsum",
        "max_length": 180, "min_length": 30, "num_beams": 6,
    },
    {
        "type": "mt5", "name": "mt5_base",
        "model_name": "google/mt5-base",
        "max_length": 150, "min_length": 20, "num_beams": 4,
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(splits: list[str], max_samples: int) -> pd.DataFrame:
    frames = []
    for split in splits:
        for name in [f"{split}-00000-of-00001.parquet", f"{split}.parquet"]:
            p = DATASET_PATH / name
            if p.exists():
                df = pd.read_parquet(p)
                df["split"] = split
                if max_samples:
                    df = df.sample(n=min(max_samples, len(df)), random_state=42)
                frames.append(df)
                logger.info("Loaded %d samples from %s", len(df), name)
                break
        else:
            logger.warning("No parquet found for split '%s'", split)

    if not frames:
        raise FileNotFoundError(f"No parquet files in {DATASET_PATH}")
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def decode_audio(audio_field) -> np.ndarray | None:
    """Convert any audio field from the parquet into a float32 16 kHz mono array."""
    from src.utils.audio_utils import AudioProcessor
    processor = AudioProcessor(target_sr=16000)
    sample = processor.process_audio_item(audio_field)
    if sample is None:
        return None
    return sample.array.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# ASR evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_asr(df: pd.DataFrame) -> pd.DataFrame:
    try:
        import jiwer
    except ImportError:
        logger.error("jiwer missing – run: pip install jiwer"); sys.exit(1)

    rows = []
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in ASR_CONFIGS:
        model_name = cfg["name"]
        logger.info("── ASR: %s ──", model_name)

        try:
            if cfg["type"] == "whisper":
                from src.asr.whisper import WhisperASR
                model = WhisperASR(cfg)
            else:
                from src.asr.wav2vec2 import Wav2Vec2ASR
                model = Wav2Vec2ASR(cfg)
            model.load_model()
        except Exception as exc:
            logger.error("Load failed %s: %s", model_name, exc)
            continue

        for idx, row in df.iterrows():
            ref = str(row.get("text", "")).strip()
            if not ref:
                continue
            audio = decode_audio(row.get("audio"))
            if audio is None:
                continue

            t0 = time.time()
            try:
                result = model.transcribe(audio, sample_rate=16000)
                hyp = result.text.strip()
                conf = float(result.confidence or 0.0)
            except Exception as exc:
                logger.warning("Transcription error idx=%s: %s", idx, exc)
                hyp, conf = "", 0.0
            elapsed = time.time() - t0

            try:
                wer = jiwer.wer(ref, hyp) if hyp else 1.0
                cer = jiwer.cer(ref, hyp) if hyp else 1.0
            except Exception:
                wer = cer = 1.0

            ref_terms = extract_medical_terms(ref)
            hyp_terms = extract_medical_terms(hyp)
            mter = 1.0 - len(ref_terms & hyp_terms) / len(ref_terms) if ref_terms else 0.0

            rows.append({
                "sample_id":   int(idx),
                "split":       row.get("split", ""),
                "duration_s":  float(row.get("duration", 0)),
                "asr_model":   model_name,
                "reference":   ref,
                "hypothesis":  hyp,
                "confidence":  round(conf, 4),
                "wer":         round(wer, 4),
                "cer":         round(cer, 4),
                "mter":        round(mter, 4),
                "asr_time_s":  round(elapsed, 3),
            })

        try:
            model.cleanup()
        except Exception:
            pass

    result_df = pd.DataFrame(rows)
    # save cache as CSV (lightweight) and full Excel
    result_df.to_csv(CACHE_DIR / "asr_transcriptions.csv", index=False)
    return result_df


# ──────────────────────────────────────────────────────────────────────────────
# AI enhancement
# ──────────────────────────────────────────────────────────────────────────────

def apply_ai_enhancement(asr_df: pd.DataFrame) -> pd.DataFrame:
    from src.pipeline.ai_enhancer import AIEnhancer
    enhancer = AIEnhancer()
    asr_df = asr_df.copy()

    if not enhancer.enabled:
        logger.warning("AI enhancement disabled (no OPENAI_API_KEY)")
        asr_df["enhanced_hypothesis"]   = asr_df["hypothesis"]
        asr_df["ai_improvement_score"]  = 0.0
        return asr_df

    enhanced, scores = [], []
    for _, row in asr_df.iterrows():
        enh = enhancer.enhance(row["hypothesis"])
        enhanced.append(enh.enhanced_text)
        scores.append(round(enh.improvement_score, 4))

    asr_df["enhanced_hypothesis"]  = enhanced
    asr_df["ai_improvement_score"] = scores
    return asr_df


# ──────────────────────────────────────────────────────────────────────────────
# Summarization evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_summarization(asr_df: pd.DataFrame) -> pd.DataFrame:
    from src.summarization.medical_summarizer import (
        BartSummarizer, MT5Summarizer, ExtractiveSummarizer, SummarizationEvaluator,
    )
    evaluator = SummarizationEvaluator()
    rows = []

    for cfg in SUMMARIZER_CONFIGS:
        sname = cfg["name"]
        logger.info("── Summarizer: %s ──", sname)
        try:
            if cfg["type"] == "extractive":
                summarizer = ExtractiveSummarizer(cfg)
            elif cfg["type"] == "bart":
                summarizer = BartSummarizer(cfg)
            else:
                summarizer = MT5Summarizer(cfg)
            summarizer.load_model()
        except Exception as exc:
            logger.error("Load failed %s: %s", sname, exc); continue

        for _, row in asr_df.iterrows():
            ref = row["reference"]
            if not ref:
                continue

            for text_col, label in [("hypothesis", "raw"), ("enhanced_hypothesis", "ai_enhanced")]:
                text = str(row.get(text_col, "")).strip()
                if not text:
                    continue
                t0 = time.time()
                try:
                    res = summarizer.summarize(text)
                    summary = res.summary
                except Exception as exc:
                    logger.warning("Summarize failed: %s", exc); summary = ""
                elapsed = time.time() - t0

                m = evaluator.evaluate(ref, summary)
                rows.append({
                    "sample_id":          row["sample_id"],
                    "asr_model":          row["asr_model"],
                    "summarizer":         sname,
                    "input_type":         label,
                    "wer":                row["wer"],
                    "mter":               row["mter"],
                    "summary":            summary,
                    "rouge1":             round(m.get("rouge1", 0), 4),
                    "rouge2":             round(m.get("rouge2", 0), 4),
                    "rougeL":             round(m.get("rougeL", 0), 4),
                    "bleu":               round(m.get("bleu", 0), 4),
                    "bert_score":         round(m.get("bert_score", 0), 4),
                    "medical_pres":       round(m.get("medical_preservation", 0), 4),
                    "compression_ratio":  round(m.get("compression_ratio", 0), 4),
                    "sum_time_s":         round(elapsed, 3),
                })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Excel writer helper
# ──────────────────────────────────────────────────────────────────────────────

def _to_excel(df: pd.DataFrame, path: Path, sheet_name: str = "Results"):
    """Write DataFrame to Excel with auto-width columns."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        ws = writer.sheets[sheet_name]
        for col_cells in ws.columns:
            max_len = max(
                len(str(cell.value)) if cell.value is not None else 0
                for cell in col_cells
            )
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 3, 60)
    logger.info("Saved → %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# Analysis & reporting
# ──────────────────────────────────────────────────────────────────────────────

def analyse_results(asr_df: pd.DataFrame, sum_df: pd.DataFrame) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. ASR comparison ─────────────────────────────────────────────────────
    asr_agg = (
        asr_df.groupby("asr_model")[["wer", "cer", "mter", "confidence", "asr_time_s"]]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    asr_agg.columns = ["_".join(c) for c in asr_agg.columns]
    asr_agg = asr_agg.reset_index()
    _to_excel(asr_agg, OUTPUT_DIR / "asr_results.xlsx", "ASR Comparison")

    # also write full per-sample data as a second sheet
    with pd.ExcelWriter(OUTPUT_DIR / "asr_results.xlsx", engine="openpyxl", mode="a") as writer:
        asr_df.drop(columns=["reference", "hypothesis", "enhanced_hypothesis"],
                    errors="ignore").to_excel(writer, sheet_name="Per-Sample", index=False)

    # ── 2. Summarisation comparison ───────────────────────────────────────────
    raw_sum = sum_df[sum_df["input_type"] == "raw"]
    enh_sum = sum_df[sum_df["input_type"] == "ai_enhanced"]

    sum_agg = (
        raw_sum.groupby(["asr_model", "summarizer"])
        [["rouge1", "rouge2", "rougeL", "bleu", "bert_score", "medical_pres"]]
        .mean()
        .round(4)
        .reset_index()
    )
    _to_excel(sum_agg, OUTPUT_DIR / "summarization_results.xlsx", "Summarizer Comparison")

    with pd.ExcelWriter(OUTPUT_DIR / "summarization_results.xlsx",
                        engine="openpyxl", mode="a") as writer:
        sum_df.drop(columns=["summary"], errors="ignore").to_excel(
            writer, sheet_name="Per-Sample", index=False
        )

    # ── 3. AI impact ──────────────────────────────────────────────────────────
    grp_cols = ["asr_model", "summarizer"]
    metric_cols = ["rouge1", "rougeL", "medical_pres"]
    r_mean = raw_sum.groupby(grp_cols)[metric_cols].mean()
    e_mean = enh_sum.groupby(grp_cols)[metric_cols].mean()
    ai_impact = (e_mean - r_mean).round(4).reset_index()
    ai_impact.columns = grp_cols + [f"delta_{c}" for c in metric_cols]
    _to_excel(ai_impact, OUTPUT_DIR / "ai_impact.xlsx", "AI Impact")

    # ── 4. Best duo ───────────────────────────────────────────────────────────
    # Score = 0.5*rouge1 + 0.3*medical_pres + 0.2*(1-wer)
    combo = sum_agg.copy()
    wer_lookup = asr_df.groupby("asr_model")["wer"].mean()
    combo["wer_mean"] = combo["asr_model"].map(wer_lookup)
    combo["composite"] = (
        0.5 * combo["rouge1"] +
        0.3 * combo["medical_pres"] +
        0.2 * (1 - combo["wer_mean"].fillna(1))
    ).round(4)

    best_row = combo.loc[combo["composite"].idxmax()]
    best_duo = {
        "best_asr_model":     best_row["asr_model"],
        "best_summarizer":    best_row["summarizer"],
        "rouge1":             float(best_row["rouge1"]),
        "medical_pres":       float(best_row["medical_pres"]),
        "wer_mean":           float(best_row["wer_mean"]),
        "composite_score":    float(best_row["composite"]),
    }
    with open(OUTPUT_DIR / "best_duo.json", "w", encoding="utf-8") as f:
        json.dump(best_duo, f, indent=2, ensure_ascii=False)

    # ── 5. Figures ────────────────────────────────────────────────────────────
    _make_figures(asr_df, sum_df, ai_impact, OUTPUT_DIR / "figures")

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info("\n%s\nASR SUMMARY\n%s\n%s", "="*60, "="*60, asr_agg.to_string(index=False))
    logger.info("\n%s\nSUMMARISER SUMMARY\n%s\n%s", "="*60, "="*60, sum_agg.to_string(index=False))
    logger.info("\n🏆 Best combo: %s + %s  (composite=%.4f)",
                best_duo["best_asr_model"], best_duo["best_summarizer"],
                best_duo["composite_score"])

    return best_duo


def _make_figures(asr_df, sum_df, ai_impact, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed – skipping figures"); return

    sns.set_theme(style="whitegrid", palette="muted")

    # 1. WER bar chart
    fig, ax = plt.subplots(figsize=(9, 4))
    asr_df.groupby("asr_model")["wer"].mean().sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("WER (lower = better)"); ax.set_title("ASR Model – Word Error Rate")
    fig.tight_layout(); fig.savefig(fig_dir / "asr_wer.png", dpi=150); plt.close(fig)

    # 2. ROUGE-1 heatmap
    pivot = (
        sum_df[sum_df["input_type"] == "raw"]
        .groupby(["asr_model", "summarizer"])["rouge1"].mean().unstack()
    )
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn", ax=ax)
        ax.set_title("ROUGE-1 – ASR × Summarizer")
        fig.tight_layout(); fig.savefig(fig_dir / "rouge1_heatmap.png", dpi=150); plt.close(fig)

    # 3. AI impact
    if "delta_rouge1" in ai_impact.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ai_impact["label"] = ai_impact["asr_model"] + "\n" + ai_impact["summarizer"]
        colors = ["green" if v > 0 else "tomato" for v in ai_impact["delta_rouge1"]]
        ax.bar(ai_impact["label"], ai_impact["delta_rouge1"], color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("ΔROUGE-1 (AI − raw)"); ax.set_title("AI Enhancement Impact")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        fig.tight_layout(); fig.savefig(fig_dir / "ai_impact.png", dpi=150); plt.close(fig)

    # 4. Medical preservation
    mp = sum_df.groupby(["summarizer", "input_type"])["medical_pres"].mean().unstack()
    if not mp.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        mp.plot(kind="bar", ax=ax)
        ax.set_ylabel("Medical Term Preservation")
        ax.set_title("Medical Preservation – Raw vs AI-Enhanced")
        ax.legend(title="Input"); plt.xticks(rotation=30, ha="right")
        fig.tight_layout(); fig.savefig(fig_dir / "medical_preservation.png", dpi=150)
        plt.close(fig)

    logger.info("Figures saved → %s", fig_dir)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Full ASR + summarizer evaluation")
    p.add_argument("--splits",   nargs="+", default=["test"])
    p.add_argument("--samples",  type=int,  default=100,
                   help="Samples per split (0 = all)")
    p.add_argument("--skip-asr", action="store_true",
                   help="Load cached asr_transcriptions.csv instead of re-running ASR")
    p.add_argument("--no-ai",    action="store_true",
                   help="Skip GPT enhancement step")
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset: %s", DATASET_PATH)
    df = load_dataset(args.splits, args.samples)
    logger.info("Samples loaded: %d", len(df))

    # ASR
    cache = CACHE_DIR / "asr_transcriptions.csv"
    if args.skip_asr and cache.exists():
        logger.info("Loading cached ASR from %s", cache)
        asr_df = pd.read_csv(cache)
    else:
        asr_df = evaluate_asr(df)

    # AI enhancement
    if not args.no_ai:
        asr_df = apply_ai_enhancement(asr_df)
    else:
        asr_df["enhanced_hypothesis"]  = asr_df["hypothesis"]
        asr_df["ai_improvement_score"] = 0.0

    # Summarization
    sum_df = evaluate_summarization(asr_df)

    # Analysis + Excel output
    best = analyse_results(asr_df, sum_df)

    print("\n" + "="*60)
    print("DONE – results in:", OUTPUT_DIR)
    print("="*60)
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
