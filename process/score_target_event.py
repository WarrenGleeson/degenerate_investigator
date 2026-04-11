#!/usr/bin/env python3
import math
import pickle
from typing import Dict, List, Tuple

import pandas as pd

from common.io_helpers import get_env, read_s3_csv, s3_client, s3_exists, write_csv_and_parquet
from process.train_ufc_winner_model import FEATURE_COLS


HEURISTIC_WEIGHTS: Dict[str, float] = {
    "reach_diff": 0.020,
    "height_diff": 0.015,
    "slpm_diff": 0.180,
    "sapm_diff": -0.150,
    "td_avg_diff": 0.110,
    "sub_avg_diff": 0.090,
    "str_acc_diff": 0.012,
    "str_def_diff": 0.012,
    "td_acc_diff": 0.008,
    "td_def_diff": 0.008,
    "wins_diff": 0.020,
    "losses_diff": -0.035,
    "recent_fights_diff": 0.015,
    "news_flag_diff": -0.020,
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def safe_num(value) -> float:
    try:
        if pd.isna(value):
            return 0.0
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return 0.0


def build_top_signals(row: pd.Series, max_items: int = 3) -> str:
    scored: List[Tuple[str, float]] = []
    for feature, weight in HEURISTIC_WEIGHTS.items():
        value = safe_num(row.get(feature, 0.0))
        contribution = value * weight
        scored.append((feature, contribution))
    top = sorted(scored, key=lambda x: abs(x[1]), reverse=True)[:max_items]
    return ", ".join(f"{name}={round(val, 3)}" for name, val in top)


def heuristic_probabilities(df: pd.DataFrame) -> pd.Series:
    scores = []
    for _, row in df.iterrows():
        score = 0.0
        for feature, weight in HEURISTIC_WEIGHTS.items():
            score += safe_num(row.get(feature, 0.0)) * weight
        score += 0.35 * (safe_num(row.get("f1_implied_prob", math.nan)) - 0.5)
        score -= 0.35 * (safe_num(row.get("f2_implied_prob", math.nan)) - 0.5)
        score = max(min(score, 4.0), -4.0)
        scores.append(sigmoid(score))
    return pd.Series(scores)


def confidence_bucket(prob: float) -> str:
    edge = abs(prob - 0.5)
    if edge >= 0.20:
        return "high"
    if edge >= 0.10:
        return "medium"
    return "low"


def model_probabilities(model, feature_cols: list[str], df: pd.DataFrame):
    proba_raw = model.predict_proba(df[feature_cols])
    classes = list(model.named_steps["clf"].classes_)
    if proba_raw.shape[1] == 1:
        only_class = int(classes[0])
        return pd.Series(1.0 if only_class == 1 else 0.0, index=df.index).values
    class_index = classes.index(1)
    return proba_raw[:, class_index]


def main() -> None:
    bucket = get_env("S3_BUCKET", "degenerative-investigator")
    region = get_env("AWS_REGION", "us-east-2")
    features_key = get_env("FEATURES_KEY", required=True)
    model_key = get_env("MODEL_KEY", "processed/ufc/model_artifacts.pkl")

    df = read_s3_csv(bucket=bucket, key=features_key, region=region)
    s3 = s3_client(region)

    scoring_method = "heuristic"
    if s3_exists(s3, bucket, model_key):
        model_bytes = s3.get_object(Bucket=bucket, Key=model_key)["Body"].read()
        artifact = pickle.loads(model_bytes)
        if isinstance(artifact, dict) and "model" in artifact:
            model = artifact["model"]
            feature_cols = artifact.get("feature_cols", FEATURE_COLS)
        else:
            model = artifact
            feature_cols = FEATURE_COLS
        proba = model_probabilities(model, feature_cols, df)
        scoring_method = "trained_model"
    else:
        proba = heuristic_probabilities(df).values

    df_out = df[["event_name", "event_slug", "fighter_1_name", "fighter_2_name", "weight_class"]].copy()
    df_out["scoring_method"] = scoring_method
    df_out["fighter_1_win_probability"] = proba
    df_out["fighter_2_win_probability"] = 1 - proba
    df_out["predicted_winner"] = df_out.apply(
        lambda row: row["fighter_1_name"] if row["fighter_1_win_probability"] >= 0.5 else row["fighter_2_name"],
        axis=1,
    )
    df_out["confidence_bucket"] = df_out["fighter_1_win_probability"].map(confidence_bucket)
    df_out["top_signals"] = df.apply(build_top_signals, axis=1)

    if "f1_implied_prob" in df.columns:
        df_out["fighter_1_market_implied_prob"] = pd.to_numeric(df["f1_implied_prob"], errors="coerce")
        df_out["fighter_1_model_market_delta"] = df_out["fighter_1_win_probability"] - df_out["fighter_1_market_implied_prob"]
    if "f2_implied_prob" in df.columns:
        df_out["fighter_2_market_implied_prob"] = pd.to_numeric(df["f2_implied_prob"], errors="coerce")
        df_out["fighter_2_model_market_delta"] = df_out["fighter_2_win_probability"] - df_out["fighter_2_market_implied_prob"]

    event_slug = str(df_out["event_slug"].iloc[0])
    csv_key = f"processed/ufc/{event_slug}_predictions.csv"
    parquet_key = f"processed/ufc/parquets/{event_slug}_predictions.parquet"
    write_csv_and_parquet(df_out, bucket=bucket, csv_key=csv_key, parquet_key=parquet_key, region=region)
    print(f"Wrote {len(df_out)} scored fights to s3://{bucket}/{csv_key}")


if __name__ == "__main__":
    main()
