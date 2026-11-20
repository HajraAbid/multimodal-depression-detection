"""Depression screening prediction for unseen participants.

Usage:
    python predict.py --features-dir <dir> [<dir> ...] [--model part2_outputs/final_model.joblib] [--out predictions.csv]

The directories must contain the participant-level feature files the saved model needs, in the same
format as the training data (one row per participant, column `participant_id`):
  - features_master.csv                     (hand-crafted features, produced by Part 1's extraction)
  - DAIC_WOZ_wav2vec2_temporal_1536.csv     (only if the model uses wav2vec2)
  - DAIC_WOZ_MentalBERT_1536.csv            (only if the model uses MentalBERT)
Output: participant_id, probability of depression, screening decision (1 = flag for clinical follow-up).
This is a screening aid, not a diagnosis.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import part2_models  # noqa: F401  (needed to un-pickle LateFusion)


def _find(dirs, fname):
    for d in dirs:
        p = Path(d) / fname
        if p.exists():
            return p
    raise FileNotFoundError(f"{fname} not found in {list(map(str, dirs))}")


def load_features(dirs, sources):
    frames = []
    for fname, cols in sources.items():
        df = pd.read_csv(_find(dirs, fname)).set_index("participant_id")
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{fname}: {len(missing)} expected columns missing, e.g. {missing[:5]}")
        frames.append(df[cols])
    return pd.concat(frames, axis=1, join="inner")


def predict(features_dirs, model_path="part2_outputs/final_model.joblib", ids=None):
    bundle = joblib.load(model_path)
    if isinstance(features_dirs, (str, Path)):
        features_dirs = [features_dirs]
    feats = load_features(features_dirs, bundle["sources"])
    if ids is not None:
        feats = feats.loc[list(ids)]
    X = feats[bundle["columns"]].values.astype(float)
    prob = bundle["model"].predict_proba(X)[:, 1]
    return pd.DataFrame({"participant_id": feats.index,
                         "probability_depressed": np.round(prob, 4),
                         "screen_positive": (prob >= bundle["threshold"]).astype(int)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", nargs="+", required=True)
    ap.add_argument("--model", default="part2_outputs/final_model.joblib")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = predict(a.features_dir, a.model)
    print(res.to_string(index=False))
    if a.out:
        res.to_csv(a.out, index=False)
        print("saved", a.out)
