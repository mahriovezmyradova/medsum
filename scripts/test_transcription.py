#!/usr/bin/env python3
"""
Transcription test – runs Whisper (tiny, base) and one Wav2Vec2 model on
5 samples from the test split and verifies WER is not 1.0 (model is working).

Run:
    python scripts/test_transcription.py
"""

from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.audio_utils import AudioProcessor
from src.medical.terminology import extract_medical_terms

DATASET  = Path("/Users/mahriovezmyradova/MultiMed_dataset/German")
N_TEST   = 5

MODELS = [
    {"type": "whisper",  "name": "whisper_tiny",  "model_size": "tiny",  "language": "de"},
    {"type": "whisper",  "name": "whisper_base",  "model_size": "base",  "language": "de"},
    {
        "type": "wav2vec2", "name": "wav2vec2_facebook",
        "model_name": "facebook/wav2vec2-large-xlsr-53-german",
        "language": "de",
    },
]


def load_samples(n: int):
    path = DATASET / "test-00000-of-00001.parquet"
    df = pd.read_parquet(path).head(n)
    processor = AudioProcessor(target_sr=16000)
    samples = []
    for _, row in df.iterrows():
        s = processor.process_audio_item(row["audio"])
        if s:
            samples.append((s.array, str(row["text"]).strip()))
    return samples


def run_model(cfg: dict, samples):
    try:
        import jiwer
    except ImportError:
        print("jiwer not installed – run: pip install jiwer"); sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Model: {cfg['name']}")
    print(f"{'='*60}")

    try:
        if cfg["type"] == "whisper":
            from src.asr.whisper import WhisperASR
            model = WhisperASR(cfg)
        else:
            from src.asr.wav2vec2 import Wav2Vec2ASR
            model = Wav2Vec2ASR(cfg)
        model.load_model()
    except Exception as exc:
        print(f"  LOAD FAILED: {exc}"); return None

    results = []
    for i, (audio, ref) in enumerate(samples):
        t0 = time.time()
        try:
            r = model.transcribe(audio, sample_rate=16000)
            hyp = r.text.strip()
        except Exception as exc:
            print(f"  [{i}] TRANSCRIBE ERROR: {exc}"); hyp = ""
        elapsed = time.time() - t0

        wer = jiwer.wer(ref, hyp) if hyp else 1.0
        ref_terms = extract_medical_terms(ref)
        hyp_terms = extract_medical_terms(hyp)
        mter = 1.0 - len(ref_terms & hyp_terms)/len(ref_terms) if ref_terms else 0.0

        print(f"\n  [{i}] WER={wer:.3f}  MTER={mter:.3f}  t={elapsed:.1f}s")
        print(f"  REF: {ref[:90]}")
        print(f"  HYP: {hyp[:90]}")

        results.append({"wer": wer, "mter": mter, "time": elapsed})

    try:
        model.cleanup()
    except Exception:
        pass

    avg_wer  = float(np.mean([r["wer"] for r in results]))
    avg_mter = float(np.mean([r["mter"] for r in results]))
    avg_time = float(np.mean([r["time"] for r in results]))
    print(f"\n  AVG  WER={avg_wer:.3f}  MTER={avg_mter:.3f}  time={avg_time:.1f}s")
    ok = avg_wer < 0.99
    print(f"  {'✅ PASS – model is producing transcriptions' if ok else '❌ FAIL – WER=1.0, model not working'}")
    return {"model": cfg["name"], "avg_wer": avg_wer, "avg_mter": avg_mter, "ok": ok}


def main():
    print("Loading test samples …")
    samples = load_samples(N_TEST)
    print(f"Loaded {len(samples)} samples\n")

    summary = []
    for cfg in MODELS:
        r = run_model(cfg, samples)
        if r:
            summary.append(r)

    print(f"\n{'='*60}")
    print("TRANSCRIPTION TEST SUMMARY")
    print(f"{'='*60}")
    all_ok = True
    for s in summary:
        status = "✅" if s["ok"] else "❌"
        print(f"  {status} {s['model']:30s}  WER={s['avg_wer']:.3f}  MTER={s['avg_mter']:.3f}")
        if not s["ok"]: all_ok = False

    if all_ok:
        print("\n✅ All models passed – ready to run full evaluation.")
    else:
        print("\n⚠️  Some models failed – check the output above.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
