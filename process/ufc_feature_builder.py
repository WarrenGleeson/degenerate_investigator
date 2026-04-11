#!/usr/bin/env python3
import math
import re

import pandas as pd

from common.io_helpers import get_env, read_s3_csv, write_csv_and_parquet


def parse_height_to_inches(value: str) -> float:
    text = str(value or "").strip()
    match = re.search(r"(\d+)'\s*(\d+)", text)
    if not match:
        return math.nan
    feet = int(match.group(1))
    inches = int(match.group(2))
    return float(feet * 12 + inches)


def parse_reach_to_inches(value: str) -> float:
    text = str(value or "").strip().replace('"', "")
    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else math.nan


def parse_percent(value: str) -> float:
    text = str(value or "").strip().replace("%", "")
    try:
        return float(text)
    except Exception:
        return math.nan


def parse_float(value: str) -> float:
    text = str(value or "").strip()
    try:
        return float(text)
    except Exception:
        return math.nan


def american_to_implied_prob(price: float) -> float:
    if pd.isna(price):
        return math.nan
    if price > 0:
        return 100.0 / (price + 100.0)
    if price < 0:
        return abs(price) / (abs(price) + 100.0)
    return math.nan


def build_fighter_lookup(df_profiles: pd.DataFrame) -> pd.DataFrame:
    df = df_profiles.copy()
    df["fighter_name_key"] = df["fighter_name"].astype(str).str.strip().str.lower()
    df["height_in"] = df["height"].map(parse_height_to_inches)
    df["reach_in"] = df["reach"].map(parse_reach_to_inches)
    for col in ["slpm", "sapm", "td_avg", "sub_avg"]:
        df[col + "_num"] = df[col].map(parse_float)
    for col in ["str_acc", "str_def", "td_acc", "td_def"]:
        df[col + "_num"] = df[col].map(parse_percent)
    return df


def build_news_flags(df_news: pd.DataFrame) -> pd.DataFrame:
    if df_news.empty:
        return pd.DataFrame(columns=["fighter_name_key", "news_flag_count", "news_summary"])
    df = df_news.copy()
    df["fighter_name_key"] = df["fighter_name"].astype(str).str.strip().str.lower()
    grouped = df.groupby("fighter_name_key", as_index=False).agg(
        news_flag_count=("label", "count"),
        news_summary=("summary", "first"),
    )
    return grouped


def build_market_lookup(df_odds: pd.DataFrame) -> pd.DataFrame:
    if df_odds.empty:
        return pd.DataFrame(columns=["event_name", "fighter_name_key", "american_price", "implied_prob"])
    df = df_odds.copy()
    df = df[df["market_key"] == "h2h"].copy()
    df["fighter_name_key"] = df["outcome_name"].astype(str).str.strip().str.lower()
    df["american_price"] = pd.to_numeric(df["american_price"], errors="coerce")
    df["implied_prob"] = df["american_price"].map(american_to_implied_prob)
    df = df.sort_values(["event_name", "fighter_name_key", "last_update"])
    return df.drop_duplicates(subset=["event_name", "fighter_name_key"], keep="last")


def main() -> None:
    bucket = get_env("S3_BUCKET", "degenerative-investigator")
    region = get_env("AWS_REGION", "us-east-2")
    card_key = get_env("EVENT_CARD_KEY", required=True)
    profiles_key = get_env("FIGHTER_PROFILES_KEY", "raw/ufc/fighters/fighter_profiles.csv")
    odds_key = get_env("ODDS_KEY", "raw/odds/current_mma_odds.csv")
    news_key = get_env("NEWS_KEY", "raw/news/fighter_recent_news.csv")

    df_card = read_s3_csv(bucket=bucket, key=card_key, region=region)
    df_profiles = build_fighter_lookup(read_s3_csv(bucket=bucket, key=profiles_key, region=region))
    try:
        df_odds = build_market_lookup(read_s3_csv(bucket=bucket, key=odds_key, region=region))
    except Exception:
        df_odds = build_market_lookup(pd.DataFrame())
    try:
        df_news = build_news_flags(read_s3_csv(bucket=bucket, key=news_key, region=region))
    except Exception:
        df_news = build_news_flags(pd.DataFrame())

    df = df_card.copy()
    df["fighter_1_key"] = df["fighter_1_name"].astype(str).str.strip().str.lower()
    df["fighter_2_key"] = df["fighter_2_name"].astype(str).str.strip().str.lower()

    fighter_1 = df_profiles.add_prefix("f1_")
    fighter_2 = df_profiles.add_prefix("f2_")
    df = df.merge(fighter_1, left_on="fighter_1_key", right_on="f1_fighter_name_key", how="left")
    df = df.merge(fighter_2, left_on="fighter_2_key", right_on="f2_fighter_name_key", how="left")

    news_1 = df_news.add_prefix("f1_")
    news_2 = df_news.add_prefix("f2_")
    df = df.merge(news_1, left_on="fighter_1_key", right_on="f1_fighter_name_key", how="left")
    df = df.merge(news_2, left_on="fighter_2_key", right_on="f2_fighter_name_key", how="left")

    odds_1 = df_odds.add_prefix("f1_")
    odds_2 = df_odds.add_prefix("f2_")
    df = df.merge(odds_1[["f1_event_name", "f1_fighter_name_key", "f1_american_price", "f1_implied_prob"]], left_on="fighter_1_key", right_on="f1_fighter_name_key", how="left")
    df = df.merge(odds_2[["f2_event_name", "f2_fighter_name_key", "f2_american_price", "f2_implied_prob"]], left_on="fighter_2_key", right_on="f2_fighter_name_key", how="left")

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
        ("news_flag_count", "news_flag_diff"),
    ]
    for base_col, diff_col in numeric_pairs:
        df[diff_col] = pd.to_numeric(df[f"f1_{base_col}"], errors="coerce") - pd.to_numeric(df[f"f2_{base_col}"], errors="coerce")

    event_slug = str(df["event_slug"].iloc[0])
    csv_key = f"processed/ufc/{event_slug}_fight_features.csv"
    parquet_key = f"processed/ufc/parquets/{event_slug}_fight_features.parquet"
    write_csv_and_parquet(df, bucket=bucket, csv_key=csv_key, parquet_key=parquet_key, region=region)
    print(f"Wrote {len(df)} matchup rows to s3://{bucket}/{csv_key}")


if __name__ == "__main__":
    main()
