#!/usr/bin/env python3
import os
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from common.io_helpers import get_env, read_s3_csv, write_csv_and_parquet


BASE_URL = "http://ufcstats.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, timeout=30, headers=HEADERS)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_profile(url: str) -> dict:
    soup = fetch_soup(url)
    name = soup.select_one(".b-content__title-highlight")
    full_name = name.get_text(" ", strip=True) if name else ""

    stat_map = {}
    for item in soup.select(".b-list__box-list-item"):
        text = item.get_text(" ", strip=True)
        if ":" in text:
            key, value = text.split(":", 1)
            stat_map[key.strip().lower().replace(" ", "_")] = value.strip()

    rows = []
    for row in soup.select("tr.b-fight-details__table-row"):
        cells = [cell.get_text(" ", strip=True) for cell in row.select("td")]
        if len(cells) < 10:
            continue
        rows.append(
            {
                "result": cells[0],
                "opponent": cells[1],
                "fighter": cells[1],
                "event": cells[6] if len(cells) > 6 else "",
                "date": cells[12] if len(cells) > 12 else "",
                "method": cells[7] if len(cells) > 7 else "",
                "round": cells[8] if len(cells) > 8 else "",
                "time": cells[9] if len(cells) > 9 else "",
            }
        )

    wins = sum(1 for row in rows if row["result"].upper() == "W")
    losses = sum(1 for row in rows if row["result"].upper() == "L")
    draws = sum(1 for row in rows if row["result"].upper() in {"D", "NC"})

    return {
        "fighter_name": full_name,
        "fighter_url": url,
        "height": stat_map.get("height", ""),
        "weight": stat_map.get("weight", ""),
        "reach": stat_map.get("reach", ""),
        "stance": stat_map.get("stance", ""),
        "dob": stat_map.get("dob", ""),
        "slpm": stat_map.get("slpm", ""),
        "str_acc": stat_map.get("str._acc.", ""),
        "sapm": stat_map.get("sapm", ""),
        "str_def": stat_map.get("str._def", ""),
        "td_avg": stat_map.get("td_avg", ""),
        "td_acc": stat_map.get("td_acc.", ""),
        "td_def": stat_map.get("td_def.", ""),
        "sub_avg": stat_map.get("sub._avg.", ""),
        "career_wins_scraped": wins,
        "career_losses_scraped": losses,
        "career_other_scraped": draws,
        "recent_fights_scraped": len(rows),
    }


def main() -> None:
    bucket = get_env("S3_BUCKET", "degenerative-investigator")
    region = get_env("AWS_REGION", "us-east-2")
    event_card_key = get_env("EVENT_CARD_KEY", required=True)
    delay = float(get_env("DELAY_BETWEEN_REQUESTS", "0.25"))

    df_card = read_s3_csv(bucket=bucket, key=event_card_key, region=region)
    urls = sorted(set(df_card["fighter_1_url"].dropna().tolist() + df_card["fighter_2_url"].dropna().tolist()))

    profiles = []
    for index, url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] {url}")
        profiles.append(parse_profile(url))
        time.sleep(delay)

    df = pd.DataFrame(profiles).drop_duplicates(subset=["fighter_url"]).sort_values("fighter_name")
    csv_key = "raw/ufc/fighters/fighter_profiles.csv"
    parquet_key = "raw/ufc/fighters/parquets/fighter_profiles.parquet"
    write_csv_and_parquet(df, bucket=bucket, csv_key=csv_key, parquet_key=parquet_key, region=region)
    print(f"Wrote {len(df)} fighter profiles to s3://{bucket}/{csv_key}")


if __name__ == "__main__":
    main()
