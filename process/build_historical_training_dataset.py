#!/usr/bin/env python3
import pandas as pd

from common.io_helpers import get_env, read_s3_csv, write_csv_and_parquet
from process.ufc_feature_builder import build_fighter_lookup


def main() -> None:
    bucket = get_env("S3_BUCKET", "degenerative-investigator")
    region = get_env("AWS_REGION", "us-east-2")
    fights_key = get_env("HISTORICAL_FIGHTS_KEY", "raw/ufc/fights/historical_fights.csv")
    profiles_key = get_env("HISTORICAL_FIGHTER_PROFILES_KEY", "raw/ufc/fighters/historical_fighter_profiles.csv")

    df_fights = read_s3_csv(bucket=bucket, key=fights_key, region=region)
    df_profiles = build_fighter_lookup(read_s3_csv(bucket=bucket, key=profiles_key, region=region))

    df = df_fights.copy()
    df["fighter_1_key"] = df["fighter_1_url"].astype(str).str.strip().str.lower()
    df["fighter_2_key"] = df["fighter_2_url"].astype(str).str.strip().str.lower()
    df_profiles["fighter_url_key"] = df_profiles["fighter_url"].astype(str).str.strip().str.lower()

    fighter_1 = df_profiles.add_prefix("f1_")
    fighter_2 = df_profiles.add_prefix("f2_")
    df = df.merge(fighter_1, left_on="fighter_1_key", right_on="f1_fighter_url_key", how="left")
    df = df.merge(fighter_2, left_on="fighter_2_key", right_on="f2_fighter_url_key", how="left")

    numeric_pairs = [
        ("height_in", "height_diff"),
        ("reach_in", "reach_diff"),
        ("slpm_num", "slpm_diff"),
        ("sapm_num", "sapm_diff"),
        ("td_avg_num", "td_avg_diff"),
        ("sub_avg_num", "sub_avg_diff"),
        ("str_acc_num", "str_acc_diff"),
        ("str_def_num", "str_def_diff"),
        ("td_acc_num", "td_acc_diff"),
        ("td_def_num", "td_def_diff"),
        ("career_wins_scraped", "wins_diff"),
        ("career_losses_scraped", "losses_diff"),
        ("recent_fights_scraped", "recent_fights_diff"),
    ]
    for base_col, diff_col in numeric_pairs:
        df[diff_col] = pd.to_numeric(df[f"f1_{base_col}"], errors="coerce") - pd.to_numeric(df[f"f2_{base_col}"], errors="coerce")

    df["news_flag_diff"] = 0.0

    keep_cols = [
        "event_name",
        "event_url",
        "fighter_1_name",
        "fighter_2_name",
        "weight_class",
        "fighter_1_win",
        "height_diff",
        "reach_diff",
        "slpm_diff",
        "sapm_diff",
        "td_avg_diff",
        "sub_avg_diff",
        "str_acc_diff",
        "str_def_diff",
        "td_acc_diff",
        "td_def_diff",
        "wins_diff",
        "losses_diff",
        "recent_fights_diff",
        "news_flag_diff",
    ]
    df_out = df[keep_cols].copy()

    csv_key = "processed/ufc/training_dataset.csv"
    parquet_key = "processed/ufc/parquets/training_dataset.parquet"
    write_csv_and_parquet(df_out, bucket=bucket, csv_key=csv_key, parquet_key=parquet_key, region=region)
    print(f"Wrote {len(df_out)} training rows to s3://{bucket}/{csv_key}")


if __name__ == "__main__":
    main()
