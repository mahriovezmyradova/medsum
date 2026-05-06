#!/usr/bin/env python3
"""
Audio loading test – verifies the dataset audio can be decoded correctly
before running any ASR model.
"""
import sys, io
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.audio_utils import AudioProcessor

DATASET = Path("/Users/mahriovezmyradova/MultiMed_dataset/German")
SPLITS  = ["test"]
N_TEST  = 10   # samples to test

def main():
    processor = AudioProcessor(target_sr=16000)
    passed = failed = 0

    for split in SPLITS:
        for fname in [f"{split}-00000-of-00001.parquet", f"{split}.parquet"]:
            path = DATASET / fname
            if path.exists(): break
        else:
            print(f"[SKIP] No parquet for split '{split}'"); continue

        df = pd.read_parquet(path).head(N_TEST)
        print(f"\n{'='*50}")
        print(f"Split: {split}  |  Testing {len(df)} samples")
        print(f"{'='*50}")

        for idx, row in df.iterrows():
            ref  = str(row.get("text","")).strip()[:80]
            dur  = float(row.get("duration", 0))
            audio_field = row.get("audio")

            sample = processor.process_audio_item(audio_field)
            if sample is None:
                print(f"  [{idx:3d}] FAIL  – decode returned None")
                failed += 1
                continue

            arr = sample.array
            # Sanity checks
            ok_shape  = arr.ndim == 1
            ok_dtype  = arr.dtype == np.float32
            ok_range  = np.max(np.abs(arr)) <= 1.01
            ok_sr     = sample.sampling_rate == 16000
            ok_dur    = abs(sample.duration - dur) < 2.0   # within 2 s

            checks = [ok_shape, ok_dtype, ok_range, ok_sr, ok_dur]
            status = "OK  " if all(checks) else "WARN"
            print(
                f"  [{idx:3d}] {status} | shape={arr.shape} sr={sample.sampling_rate} "
                f"dur={sample.duration:.1f}s range=[{arr.min():.2f},{arr.max():.2f}] "
                f"| ref: \"{ref}...\""
            )
            if all(checks): passed += 1
            else:
                failed += 1
                print(f"         checks: shape={ok_shape} dtype={ok_dtype} "
                      f"range={ok_range} sr={ok_sr} dur={ok_dur}")

    print(f"\n{'='*50}")
    print(f"Audio test complete: {passed} passed / {failed} failed")
    if failed == 0:
        print("✅ All audio samples decoded correctly – ready for ASR tests.")
    else:
        print("⚠️  Some samples failed – check AudioProcessor or dataset path.")
    return failed

if __name__ == "__main__":
    sys.exit(main())
