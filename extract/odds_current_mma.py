#!/usr/bin/env python3
import os
from datetime import datetime, timezone

import pandas as pd
import requests

from common.io_helpers import get_env, slugify, write_csv_and_parquet


def american_to_decimal(price: float) -> float:
    if price > 0:
        return round((price / 100.0) + 1.0, 4)
    if price < 0:
        return round((100.0 / abs(price)) + 1.0, 4)
    return 0.0


def fetch_the_odds_api(api_key: str, regions: str, markets: str, bookmakers: str) -> list[dict]:
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if bookmakers:
        params["bookmakers"] = bookmakers

    response = requests.get(
        "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds",
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def normalize_odds(payload: list[dict]) -> pd.DataFrame:
    rows = []
    extracted_at = datetime.now(timezone.utc).isoformat()

    for event in payload:
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        for bookmaker in event.get("bookmakers", []):
            bookmaker_key = bookmaker.get("key", "")
            bookmaker_title = bookmaker.get("title", "")
            updated_at = bookmaker.get("last_update", "")
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    if price is None:
                        continue
                    rows.append(
                        {
                            "event_id": event.get("id", ""),
                            "event_name": event.get("home_team") + " vs " + event.get("away_team"),
                            "event_slug": slugify(event.get("home_team", "") + "-vs-" + event.get("away_team", "")),
                            "commence_time": event.get("commence_time", ""),
                            "fighter_1_name": home_team,
                            "fighter_2_name": away_team,
                            "bookmaker_key": bookmaker_key,
                            "bookmaker_title": bookmaker_title,
                            "market_key": market_key,
                            "outcome_name": outcome.get("name", ""),
                            "american_price": price,
                            "decimal_price": american_to_decimal(float(price)),
                            "last_update": updated_at,
                            "extracted_at": extracted_at,
                        }
                    )

    if not rows:
        raise RuntimeError("No odds rows returned. Check ODDS_API_KEY or bookmaker filters.")
    return pd.DataFrame(rows)


def main() -> None:
    bucket = get_env("S3_BUCKET", "degenerative-investigator")
    region = get_env("AWS_REGION", "us-east-2")
    api_key = get_env("ODDS_API_KEY", required=True)
    bookmaker_filter = get_env("BOOKMAKERS", "")
    regions = get_env("ODDS_REGIONS", "us,eu,uk")
    markets = get_env("ODDS_MARKETS", "h2h")

    payload = fetch_the_odds_api(api_key=api_key, regions=regions, markets=markets, bookmakers=bookmaker_filter)
    df = normalize_odds(payload)
    csv_key = "raw/odds/current_mma_odds.csv"
    parquet_key = "raw/odds/parquets/current_mma_odds.parquet"
    write_csv_and_parquet(df, bucket=bucket, csv_key=csv_key, parquet_key=parquet_key, region=region)
    print(f"Wrote {len(df)} odds rows to s3://{bucket}/{csv_key}")


if __name__ == "__main__":
    main()
