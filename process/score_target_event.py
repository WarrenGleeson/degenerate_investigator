#!/usr/bin/env python3
import pickle

import pandas as pd

from common.io_helpers import get_env, read_s3_csv, s3_client, write_csv_and_parquet
from process.train_ufc_winner_model import FEATURE_COLS


def main() -> None:
    bucket = get_env("S3_BUCKET", "degenerative-investigator")
    region = get_env("AWS_REGION", "us-east-2")
    features_key = get_env("FEATURES_KEY", required=True)
    model_key = get_env("MODEL_KEY", "processed/ufc/model_artifacts.pkl")

    df = read_s3_csv(bucket=bucket, key=features_key, region=region)
    s3 = s3_client(region)
    model_bytes = s3.get_object(Bucket=bucket, Key=model_key)["Body"].read()
    model = pickle.loads(model_bytes)

    proba = model.predict_proba(df[FEATURE_COLS])[:, 1]
    df_out = df[["event_name", "event_slug", "fighter_1_name", "fighter_2_name", "weight_class"]].copy()
    df_out["fighter_1_win_probability"] = proba
    df_out["fighter_2_win_probability"] = 1 - proba
    df_out["predicted_winner"] = df_out.apply(
        lambda row: row["fighter_1_name"] if row["fighter_1_win_probability"] >= 0.5 else row["fighter_2_name"],
        axis=1,
    )
    if "f1_implied_prob" in df.columns:
        df_out["fighter_1_market_implied_prob"] = df["f1_implied_prob"]
    if "f2_implied_prob" in df.columns:
        df_out["fighter_2_market_implied_prob"] = df["f2_implied_prob"]
    if "f1_implied_prob" in df.columns:
        df_out["fighter_1_model_market_delta"] = df_out["fighter_1_win_probability"] - pd.to_numeric(df["f1_implied_prob"], errors="coerce")
    if "f2_implied_prob" in df.columns:
        df_out["fighter_2_model_market_delta"] = df_out["fighter_2_win_probability"] - pd.to_numeric(df["f2_implied_prob"], errors="coerce")

    event_slug = str(df_out["event_slug"].iloc[0])
    csv_key = f"processed/ufc/{event_slug}_predictions.csv"
    parquet_key = f"processed/ufc/parquets/{event_slug}_predictions.parquet"
    write_csv_and_parquet(df_out, bucket=bucket, csv_key=csv_key, parquet_key=parquet_key, region=region)
    print(f"Wrote {len(df_out)} scored fights to s3://{bucket}/{csv_key}")


if __name__ == "__main__":
    main()
