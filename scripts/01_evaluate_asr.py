#!/usr/bin/env python3
"""
Script 01 – Full ASR Evaluation
================================
Evaluates all five ASR models on the MultiMed German dataset and produces
a comprehensive, multi-sheet Excel workbook plus publication-quality figures.

Metrics computed per sample and model
--------------------------------------
  WER   – Word Error Rate
  CER   – Character Error Rate
  MER   – Match Error Rate
  WIL   – Word Information Lost
  MTER  – Medical Term Error Rate  (domain-specific, using our terminology DB)
  SUB%  – Substitution rate
  DEL%  – Deletion rate
  INS%  – Insertion rate
  CONF  – Model confidence score
  RTF   – Real-Time Factor  (processing time / audio duration)

Excel workbook sheets
----------------------
  1. Summary        – mean ± std for every metric per model, ranked
  2. Per-Sample     – every single transcription with all metrics
  3. Statistical    – pairwise significance tests (Wilcoxon signed-rank)
  4. Medical-Terms  – per-category medical term error rates
  5. Duration-Bins  – WER bucketed by audio length (short / medium / long)
  6. Error-Types    – substitution / deletion / insertion breakdown per model

Figures (data/outputs/asr_evaluation/figures/)
-----------------------------------------------
  wer_boxplot.png          – WER distribution per model
  wer_cer_comparison.png   – WER vs CER grouped bar chart
  error_type_breakdown.png – stacked bar: sub / del / ins per model
  mter_by_category.png     – medical term error rates per category
  wer_vs_duration.png      – scatter + regression: WER ~ audio length
  rtf_comparison.png       – real-time factor per model
  confidence_hist.png      – confidence score distributions

Checkpointing
-------------
  Results are saved every 25 samples so a crash loses at most 25 samples.
  Re-running the script appends only the missing samples.

Usage
-----
  # Full test split (recommended for diploma thesis)
  python scripts/01_evaluate_asr.py

  # All three splits
  python scripts/01_evaluate_asr.py --splits train eval test

  # Quick smoke-test (50 samples)
  python scripts/01_evaluate_asr.py --samples 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── project path ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.audio_utils import AudioProcessor
from src.medical.terminology import MEDICAL_TERMS, extract_medical_terms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATASET_PATH = Path(os.getenv("MULTIMED_PATH", "/Users/mahriovezmyradova/MultiMed_dataset/German"))

OUTPUT_DIR  = ROOT / "data" / "outputs" / "asr_evaluation"
FIGURES_DIR = OUTPUT_DIR / "figures"
CACHE_PATH  = OUTPUT_DIR / "cache" / "asr_raw.csv"

CHECKPOINT_EVERY = 25   # save partial results every N samples per model

ASR_MODELS = [
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

# Duration buckets for WER-by-length analysis
DURATION_BINS   = [0, 5, 10, 20, 60]
DURATION_LABELS = ["0-5 s", "5-10 s", "10-20 s", "20+ s"]


# ──────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(splits: list[str], max_samples: int) -> pd.DataFrame:
    processor = AudioProcessor(target_sr=16000)
    frames = []

    for split in splits:
        for fname in [f"{split}-00000-of-00001.parquet", f"{split}.parquet"]:
            p = DATASET_PATH / fname
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            df["split"] = split
            if max_samples:
                df = df.sample(n=min(max_samples, len(df)), random_state=42)
            df = df.reset_index(drop=True)
            frames.append(df)
            log.info("Loaded %d samples from %s/%s", len(df), split, fname)
            break
        else:
            log.warning("No parquet found for split '%s' in %s", split, DATASET_PATH)

    if not frames:
        raise FileNotFoundError(f"No parquet files found in {DATASET_PATH}")

    combined = pd.concat(frames, ignore_index=True)
    log.info("Total samples: %d", len(combined))
    return combined


def decode_audio(audio_field, processor: AudioProcessor) -> np.ndarray | None:
    sample = processor.process_audio_item(audio_field)
    return sample.array if sample else None


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

import re as _re

def _normalise(text: str) -> str:
    """Lowercase + strip punctuation for fair WER/CER comparison.

    Whisper adds capitalisation and punctuation; the MultiMed reference is raw
    lowercase without punctuation.  Normalising both sides before metric
    computation prevents artificial inflation of CER/WER.
    German umlauts (ä,ö,ü,ß) and word characters are preserved.
    """
    text = text.lower()
    text = _re.sub(r"[^\w\s]", " ", text)   # drop punctuation
    text = _re.sub(r"\s+", " ", text).strip()
    return text


def compute_jiwer_metrics(ref: str, hyp: str) -> dict:
    """WER / CER / MER / WIL and error type breakdown via jiwer."""
    import jiwer

    empty = {
        "wer": 1.0, "cer": 1.0, "mer": 1.0, "wil": 1.0,
        "sub_rate": 0.0, "del_rate": 0.0, "ins_rate": 0.0,
        "n_ref_words": len(ref.split()),
    }
    if not hyp or not ref:
        return empty

    # Normalise both sides so capitalisation / punctuation don't inflate metrics
    ref_n = _normalise(ref)
    hyp_n = _normalise(hyp)
    if not ref_n or not hyp_n:
        return empty

    try:
        wer  = jiwer.wer(ref_n, hyp_n)
        cer  = jiwer.cer(ref_n, hyp_n)
        mer  = jiwer.mer(ref_n, hyp_n)
        wil  = jiwer.wil(ref_n, hyp_n)

        # error type breakdown
        out = jiwer.process_words(ref_n, hyp_n)
        total_ops = out.substitutions + out.deletions + out.insertions + out.hits
        total_ops = max(total_ops, 1)

        return {
            "wer":        round(min(wer, 1.0), 4),
            "cer":        round(min(cer, 1.0), 4),
            "mer":        round(min(mer, 1.0), 4),
            "wil":        round(min(wil, 1.0), 4),
            "sub_rate":   round(out.substitutions / total_ops, 4),
            "del_rate":   round(out.deletions     / total_ops, 4),
            "ins_rate":   round(out.insertions    / total_ops, 4),
            "n_ref_words": len(ref.split()),
        }
    except Exception as exc:
        log.debug("jiwer failed: %s", exc)
        return empty


def compute_mter(ref: str, hyp: str) -> dict:
    """
    Medical Term Error Rate – overall and per category.
    Returns a dict with keys:
        mter_overall, mter_symptoms, mter_diagnosis, mter_medication,
        mter_treatment, mter_diagnostics, mter_body_parts, mter_vitals,
        n_medical_terms_ref, n_medical_terms_hyp
    """
    result = {"mter_overall": 0.0, "n_medical_terms_ref": 0, "n_medical_terms_hyp": 0}
    for cat in MEDICAL_TERMS:
        result[f"mter_{cat}"] = 0.0

    ref_lower = ref.lower()
    hyp_lower = hyp.lower()

    ref_all = extract_medical_terms(ref)
    hyp_all = extract_medical_terms(hyp)
    result["n_medical_terms_ref"] = len(ref_all)
    result["n_medical_terms_hyp"] = len(hyp_all)

    if ref_all:
        result["mter_overall"] = round(1.0 - len(ref_all & hyp_all) / len(ref_all), 4)

    for cat, terms in MEDICAL_TERMS.items():
        ref_cat = {t for t in terms if t in ref_lower}
        hyp_cat = {t for t in terms if t in hyp_lower}
        if ref_cat:
            result[f"mter_{cat}"] = round(1.0 - len(ref_cat & hyp_cat) / len(ref_cat), 4)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# ASR runner
# ──────────────────────────────────────────────────────────────────────────────

def run_asr_model(cfg: dict, df: pd.DataFrame, processor: AudioProcessor) -> pd.DataFrame:
    """Transcribe all rows in df with one ASR model; return per-sample result rows."""
    model_name = cfg["name"]
    log.info("Loading model: %s", model_name)

    try:
        if cfg["type"] == "whisper":
            from src.asr.whisper import WhisperASR
            model = WhisperASR(cfg)
        else:
            from src.asr.wav2vec2 import Wav2Vec2ASR
            model = Wav2Vec2ASR(cfg)
        model.load_model()
    except Exception as exc:
        log.error("Failed to load %s: %s", model_name, exc)
        return pd.DataFrame()

    # ── load existing cache for this model ───────────────────────────────────
    cached_df = pd.DataFrame()
    done_ids: set[int] = set()
    if CACHE_PATH.exists():
        full_cache = pd.read_csv(CACHE_PATH)
        cached_df  = full_cache[full_cache["asr_model"] == model_name].copy()
        done_ids   = set(cached_df["sample_id"].astype(int))
        log.info("  %d samples already cached for %s – skipping those",
                 len(done_ids), model_name)

    rows = []
    todo = df[~df.index.isin(done_ids)]
    log.info("  Transcribing %d samples …", len(todo))

    for idx, row in tqdm(todo.iterrows(), total=len(todo), desc=model_name, ncols=80):
        ref = str(row.get("text", "")).strip()
        dur = float(row.get("duration", 0.0))

        audio = decode_audio(row.get("audio"), processor)
        if audio is None:
            continue

        t0 = time.time()
        try:
            result = model.transcribe(audio, sample_rate=16000)
            hyp  = result.text.strip()
            conf = float(result.confidence or 0.0)
        except Exception as exc:
            log.debug("Transcription error idx=%s: %s", idx, exc)
            hyp, conf = "", 0.0
        elapsed = time.time() - t0

        jw   = compute_jiwer_metrics(ref, hyp)
        mter = compute_mter(ref, hyp)

        rtf = elapsed / dur if dur > 0 else 0.0

        row_data = {
            "sample_id":       int(idx),
            "split":           str(row.get("split", "")),
            "duration_s":      round(dur, 2),
            "dur_bin":         pd.cut([dur], bins=DURATION_BINS,
                                      labels=DURATION_LABELS)[0],
            "asr_model":       model_name,
            "reference":       ref,
            "reference_norm":  _normalise(ref),
            "hypothesis":      hyp,
            "hypothesis_norm": _normalise(hyp),
            "confidence":      round(conf, 4),
            "rtf":             round(rtf, 4),
            "asr_time_s":      round(elapsed, 3),
            **jw,
            **mter,
        }
        rows.append(row_data)

        # ── checkpoint ───────────────────────────────────────────────────────
        if len(rows) % CHECKPOINT_EVERY == 0:
            _append_cache(pd.DataFrame(rows[-CHECKPOINT_EVERY:]))

    # flush remaining new rows
    if rows:
        remainder = rows[-(len(rows) % CHECKPOINT_EVERY) or len(rows):]
        if remainder:
            _append_cache(pd.DataFrame(remainder))

    try:
        model.cleanup()
    except Exception:
        pass

    # Return cached rows merged with any newly transcribed rows so the
    # summary always includes the full picture for this model.
    new_df = pd.DataFrame(rows)
    if not cached_df.empty and not new_df.empty:
        return pd.concat([cached_df, new_df], ignore_index=True)
    if not cached_df.empty:
        return cached_df
    return new_df


def _append_cache(df: pd.DataFrame):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CACHE_PATH.exists():
        df.to_csv(CACHE_PATH, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(CACHE_PATH, index=False, encoding="utf-8-sig")


# ──────────────────────────────────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────────────────────────────────

CORE_METRICS = ["wer", "cer", "mer", "wil", "mter_overall",
                "sub_rate", "del_rate", "ins_rate", "confidence", "rtf"]


def build_summary_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Sheet 1 – aggregate metrics per model, ranked by WER."""
    agg = {}
    for m in CORE_METRICS:
        if m not in df.columns:
            continue
        g = df.groupby("asr_model")[m]
        agg[f"{m}_mean"]   = g.mean().round(4)
        agg[f"{m}_std"]    = g.std().round(4)
        agg[f"{m}_median"] = g.median().round(4)
        agg[f"{m}_p95"]    = g.quantile(0.95).round(4)
        agg[f"{m}_min"]    = g.min().round(4)
        agg[f"{m}_max"]    = g.max().round(4)

    summary = pd.DataFrame(agg)

    # 95% CI for WER
    counts = df.groupby("asr_model")["wer"].count()
    stds   = df.groupby("asr_model")["wer"].std()
    summary["wer_ci95_lower"] = (summary["wer_mean"] - 1.96 * stds / np.sqrt(counts)).round(4)
    summary["wer_ci95_upper"] = (summary["wer_mean"] + 1.96 * stds / np.sqrt(counts)).round(4)
    summary["n_samples"]      = counts

    # rank
    summary = summary.sort_values("wer_mean").reset_index()
    summary.insert(0, "rank", range(1, len(summary) + 1))
    return summary


def build_statistical_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Sheet 3 – pairwise Wilcoxon signed-rank tests on WER."""
    from scipy.stats import wilcoxon
    models = df["asr_model"].unique().tolist()
    rows = []
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            wer1 = df[df["asr_model"] == m1].set_index("sample_id")["wer"]
            wer2 = df[df["asr_model"] == m2].set_index("sample_id")["wer"]
            common = wer1.index.intersection(wer2.index)
            if len(common) < 5:
                continue
            d1, d2 = wer1[common].values, wer2[common].values
            try:
                stat, p = wilcoxon(d1, d2)
                mean_diff = float(np.mean(d1) - np.mean(d2))
            except Exception:
                stat, p, mean_diff = np.nan, np.nan, np.nan

            rows.append({
                "model_A":   m1,
                "model_B":   m2,
                "mean_wer_A": round(float(np.mean(d1)), 4),
                "mean_wer_B": round(float(np.mean(d2)), 4),
                "mean_diff (A−B)": round(mean_diff, 4),
                "wilcoxon_stat": round(float(stat), 4) if not np.isnan(stat) else "",
                "p_value":   round(float(p), 6)  if not np.isnan(p)   else "",
                "significant (p<0.05)": "YES" if (not np.isnan(p) and p < 0.05) else "NO",
                "n_samples": len(common),
            })
    return pd.DataFrame(rows)


def build_medical_terms_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Sheet 4 – MTER per category per model."""
    cat_cols = [c for c in df.columns if c.startswith("mter_")]
    if not cat_cols:
        return pd.DataFrame()
    agg = df.groupby("asr_model")[cat_cols].mean().round(4).reset_index()
    # rename for readability
    agg.columns = ["asr_model"] + [c.replace("mter_", "MTER_") for c in cat_cols]
    return agg.sort_values("MTER_overall")


def build_duration_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Sheet 5 – WER / MTER bucketed by audio duration."""
    if "dur_bin" not in df.columns:
        return pd.DataFrame()
    rows = []
    for model in df["asr_model"].unique():
        sub = df[df["asr_model"] == model]
        for bucket in DURATION_LABELS:
            b = sub[sub["dur_bin"] == bucket]
            if len(b) == 0:
                continue
            rows.append({
                "asr_model":   model,
                "duration_bin": bucket,
                "n_samples":   len(b),
                "wer_mean":    round(b["wer"].mean(), 4),
                "wer_std":     round(b["wer"].std(), 4),
                "mter_mean":   round(b["mter_overall"].mean(), 4),
                "cer_mean":    round(b["cer"].mean(), 4),
            })
    return pd.DataFrame(rows)


def build_error_types_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """Sheet 6 – substitution / deletion / insertion rates."""
    cols = ["sub_rate", "del_rate", "ins_rate"]
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    return df.groupby("asr_model")[available].mean().round(4).reset_index()


# ──────────────────────────────────────────────────────────────────────────────
# Excel writer
# ──────────────────────────────────────────────────────────────────────────────

def write_excel(
    all_df: pd.DataFrame,
    summary: pd.DataFrame,
    stats: pd.DataFrame,
    med_terms: pd.DataFrame,
    dur_bins: pd.DataFrame,
    err_types: pd.DataFrame,
    path: Path,
):
    log.info("Writing Excel workbook → %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # per-sample sheet: drop raw audio, keep only analysis columns
    per_sample = all_df.drop(columns=["reference", "hypothesis"], errors="ignore")

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        _write_sheet(xw, summary,    "1_Summary",        freeze="B2")
        _write_sheet(xw, per_sample, "2_Per-Sample",     freeze="A2")
        _write_sheet(xw, stats,      "3_Statistical",    freeze="A2")
        _write_sheet(xw, med_terms,  "4_Medical-Terms",  freeze="B2")
        _write_sheet(xw, dur_bins,   "5_Duration-Bins",  freeze="A2")
        _write_sheet(xw, err_types,  "6_Error-Types",    freeze="A2")

        # full transcription dump (separate sheet for qualitative analysis)
        trans_cols = ["sample_id", "split", "duration_s", "asr_model",
                      "reference", "hypothesis", "wer", "mter_overall"]
        trans = all_df[[c for c in trans_cols if c in all_df.columns]]
        _write_sheet(xw, trans, "7_Transcriptions", freeze="A2")

    log.info("Excel workbook saved.")


def _write_sheet(writer, df: pd.DataFrame, name: str, freeze: str = "A2"):
    if df is None or df.empty:
        return
    df.to_excel(writer, sheet_name=name, index=False)
    ws = writer.sheets[name]
    if freeze:
        ws.freeze_panes = freeze
    for col in ws.columns:
        max_w = max((len(str(c.value)) if c.value else 0) for c in col)
        ws.column_dimensions[col[0].column_letter].width = min(max_w + 3, 55)


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def make_figures(df: pd.DataFrame, summary: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats as scipy_stats

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    model_order = summary["asr_model"].tolist()
    palette = sns.color_palette("muted", len(model_order))

    # ── 1. WER boxplot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="asr_model", y="wer", order=model_order,
                palette=palette, ax=ax, fliersize=2)
    ax.set_xlabel("ASR Model")
    ax.set_ylabel("Word Error Rate (WER)")
    ax.set_title("WER Distribution Across ASR Models\n(MultiMed German – test split)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "wer_boxplot.png", dpi=180)
    plt.close(fig)
    log.info("  wer_boxplot.png")

    # ── 2. WER + CER grouped bar ──────────────────────────────────────────────
    plot_df = summary[["asr_model", "wer_mean", "cer_mean"]].copy()
    plot_df.columns = ["Model", "WER", "CER"]
    plot_df = plot_df.melt(id_vars="Model", var_name="Metric", value_name="Value")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="Model", y="Value", hue="Metric",
                palette=["#2E86C1", "#E74C3C"], ax=ax, order=model_order)
    ax.set_title("WER and CER per ASR Model (mean ± 95 % CI)")
    ax.set_ylabel("Error Rate")
    ax.set_xlabel("ASR Model")
    plt.xticks(rotation=20, ha="right")
    ax.legend(title="Metric")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "wer_cer_comparison.png", dpi=180)
    plt.close(fig)
    log.info("  wer_cer_comparison.png")

    # ── 3. Error type stacked bar ─────────────────────────────────────────────
    err_cols = [c for c in ["sub_rate", "del_rate", "ins_rate"] if c in df.columns]
    if err_cols:
        err_agg = df.groupby("asr_model")[err_cols].mean().loc[model_order]
        err_agg.columns = ["Substitution", "Deletion", "Insertion"]
        fig, ax = plt.subplots(figsize=(10, 5))
        err_agg.plot(kind="bar", stacked=True, ax=ax,
                     color=["#2980B9", "#E74C3C", "#27AE60"])
        ax.set_title("Error Type Breakdown per ASR Model\n(Substitution / Deletion / Insertion)")
        ax.set_ylabel("Fraction of all operations")
        ax.set_xlabel("ASR Model")
        plt.xticks(rotation=20, ha="right")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "error_type_breakdown.png", dpi=180)
        plt.close(fig)
        log.info("  error_type_breakdown.png")

    # ── 4. MTER per category heatmap ─────────────────────────────────────────
    cat_cols = [c for c in df.columns if c.startswith("mter_") and c != "mter_overall"]
    if cat_cols:
        mter_agg = df.groupby("asr_model")[cat_cols].mean()
        mter_agg = mter_agg.loc[[m for m in model_order if m in mter_agg.index]]
        mter_agg.columns = [c.replace("mter_", "") for c in mter_agg.columns]
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(mter_agg.T, annot=True, fmt=".3f", cmap="YlOrRd",
                    linewidths=0.4, ax=ax, vmin=0, vmax=1)
        ax.set_title("Medical Term Error Rate (MTER) by Category and ASR Model")
        ax.set_xlabel("ASR Model")
        ax.set_ylabel("Medical Category")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "mter_by_category.png", dpi=180)
        plt.close(fig)
        log.info("  mter_by_category.png")

    # ── 5. WER vs audio duration scatter ─────────────────────────────────────
    fig, axes = plt.subplots(1, len(model_order), figsize=(4 * len(model_order), 4),
                              sharey=True)
    if len(model_order) == 1:
        axes = [axes]
    for ax, model, color in zip(axes, model_order, palette):
        sub = df[df["asr_model"] == model][["duration_s", "wer"]].dropna()
        ax.scatter(sub["duration_s"], sub["wer"], alpha=0.25, s=8, color=color)
        if len(sub) > 3:
            slope, intercept, r, p, _ = scipy_stats.linregress(sub["duration_s"], sub["wer"])
            x_range = np.linspace(sub["duration_s"].min(), sub["duration_s"].max(), 100)
            ax.plot(x_range, slope * x_range + intercept, color="black", lw=1.5,
                    label=f"r={r:.2f}, p={p:.3f}")
            ax.legend(fontsize=7)
        ax.set_title(model, fontsize=9)
        ax.set_xlabel("Duration (s)")
    axes[0].set_ylabel("WER")
    fig.suptitle("WER vs Audio Duration per ASR Model")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "wer_vs_duration.png", dpi=180)
    plt.close(fig)
    log.info("  wer_vs_duration.png")

    # ── 6. RTF (real-time factor) bar chart ───────────────────────────────────
    if "rtf" in df.columns:
        rtf_agg = df.groupby("asr_model")["rtf"].median().loc[model_order]
        fig, ax = plt.subplots(figsize=(9, 4))
        rtf_agg.plot(kind="bar", ax=ax, color=palette)
        ax.axhline(1.0, color="red", linestyle="--", lw=1, label="Real-time (RTF=1)")
        ax.set_title("Real-Time Factor per ASR Model (median)\nRTF < 1 = faster than real time")
        ax.set_ylabel("RTF (processing time / audio duration)")
        ax.set_xlabel("ASR Model")
        plt.xticks(rotation=20, ha="right")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "rtf_comparison.png", dpi=180)
        plt.close(fig)
        log.info("  rtf_comparison.png")

    # ── 7. Confidence histograms ──────────────────────────────────────────────
    if "confidence" in df.columns:
        fig, axes = plt.subplots(1, len(model_order), figsize=(3.5 * len(model_order), 3.5),
                                  sharey=True)
        if len(model_order) == 1:
            axes = [axes]
        for ax, model, color in zip(axes, model_order, palette):
            sub = df[df["asr_model"] == model]["confidence"].dropna()
            ax.hist(sub, bins=30, color=color, alpha=0.8, edgecolor="white")
            ax.axvline(sub.median(), color="black", lw=1.5,
                       label=f"median={sub.median():.2f}")
            ax.set_title(model, fontsize=9)
            ax.set_xlabel("Confidence")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("Count")
        fig.suptitle("Model Confidence Score Distributions")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "confidence_hist.png", dpi=180)
        plt.close(fig)
        log.info("  confidence_hist.png")

    # ── 8. Overall ranking summary (final figure) ─────────────────────────────
    rank_metrics = {
        "WER↓":   "wer_mean",
        "CER↓":   "cer_mean",
        "MTER↓":  "mter_overall" if "mter_overall" in df.columns else None,
        "RTF↓":   "rtf" if "rtf" in df.columns else None,
    }
    rank_data = {}
    for label, col in rank_metrics.items():
        if col is None:
            continue
        if col in summary.columns:
            rank_data[label] = summary.set_index("asr_model")[col]
        elif col in df.columns:
            rank_data[label] = df.groupby("asr_model")[col].mean()

    if rank_data:
        rank_df = pd.DataFrame(rank_data).loc[model_order]
        # normalise each column 0→1 (lower is better for all metrics here)
        norm = (rank_df - rank_df.min()) / (rank_df.max() - rank_df.min() + 1e-9)
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(norm.columns))
        w = 0.15
        for i, (model, color) in enumerate(zip(model_order, palette)):
            ax.bar(x + i * w, norm.loc[model], width=w, label=model, color=color)
        ax.set_xticks(x + w * (len(model_order) - 1) / 2)
        ax.set_xticklabels(norm.columns)
        ax.set_ylabel("Normalised score (0=best, 1=worst)")
        ax.set_title("Multi-Metric Comparison – Normalised Scores")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "multi_metric_ranking.png", dpi=180)
        plt.close(fig)
        log.info("  multi_metric_ranking.png")

    log.info("All figures saved → %s", FIGURES_DIR)


# ──────────────────────────────────────────────────────────────────────────────
# Console summary
# ──────────────────────────────────────────────────────────────────────────────

def print_console_summary(summary: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  ASR EVALUATION RESULTS  –  MultiMed German")
    print("=" * 70)
    cols = ["rank", "asr_model", "n_samples",
            "wer_mean", "wer_std", "wer_ci95_lower", "wer_ci95_upper",
            "cer_mean", "mter_overall" if "mter_overall" in summary.columns else "mter_overall_mean",
            "rtf_median" if "rtf_median" in summary.columns else "rtf_mean"]
    available = [c for c in cols if c in summary.columns]
    print(summary[available].to_string(index=False))
    print("=" * 70)
    best = summary.iloc[0]["asr_model"]
    best_wer = summary.iloc[0]["wer_mean"]
    print(f"\n  ✅ Best ASR model: {best}  (mean WER = {best_wer:.4f})")
    print(f"  Results → {OUTPUT_DIR}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Full ASR evaluation on MultiMed German")
    p.add_argument("--splits",  nargs="+", default=["test"],
                   help="Dataset splits to evaluate (default: test)")
    p.add_argument("--samples", type=int, default=0,
                   help="Max samples per split. 0 = all (default)")
    p.add_argument("--models",  nargs="+",
                   default=[m["name"] for m in ASR_MODELS],
                   help="Subset of model names to run")
    p.add_argument("--no-cache", action="store_true",
                   help="Ignore existing cache and re-run everything")
    return p.parse_args()


def main():
    args = parse_args()

    if args.no_cache and CACHE_PATH.exists():
        CACHE_PATH.unlink()
        log.info("Cache cleared.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # load dataset
    df = load_dataset(args.splits, args.samples)
    processor = AudioProcessor(target_sr=16000)

    # run models
    selected = [m for m in ASR_MODELS if m["name"] in args.models]
    all_rows = []
    for cfg in selected:
        part = run_asr_model(cfg, df, processor)
        if not part.empty:
            all_rows.append(part)

    if not all_rows:
        log.error("No results produced – check model configs and dataset path.")
        sys.exit(1)

    all_df = pd.concat(all_rows, ignore_index=True)

    # build analysis sheets
    summary   = build_summary_sheet(all_df)
    stats     = build_statistical_sheet(all_df)
    med_terms = build_medical_terms_sheet(all_df)
    dur_bins  = build_duration_sheet(all_df)
    err_types = build_error_types_sheet(all_df)

    # write Excel
    write_excel(
        all_df, summary, stats, med_terms, dur_bins, err_types,
        path=OUTPUT_DIR / "asr_full_results.xlsx",
    )

    # write figures
    make_figures(all_df, summary)

    # console summary
    print_console_summary(summary)

    # also dump clean CSV for script 02
    all_df.to_csv(OUTPUT_DIR / "asr_full_results.csv", index=False, encoding="utf-8-sig")
    log.info("CSV dump → %s/asr_full_results.csv", OUTPUT_DIR)


if __name__ == "__main__":
    main()
