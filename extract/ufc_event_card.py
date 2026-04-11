#!/usr/bin/env python3
import os
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from common.io_helpers import get_env, slugify, write_csv_and_parquet


BASE_URL = "http://ufcstats.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, timeout=30, headers=HEADERS)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def parse_event_card(event_url: str) -> tuple[pd.DataFrame, str]:
    soup = fetch_soup(event_url)

    title_node = soup.select_one(".b-content__title-highlight") or soup.select_one(".b-content__title")
    event_name = title_node.get_text(" ", strip=True) if title_node else event_url
    event_slug = slugify(event_name)

    meta_text = " ".join(x.get_text(" ", strip=True) for x in soup.select(".b-list__box-list-item"))

    rows = []
    for row in soup.select("tr.b-fight-details__table-row"):
        links = row.select("a")
        fighter_links = [a for a in links if "/fighter-details/" in (a.get("href") or "")]
        bout_link = next((a.get("href") for a in links if "/fight-details/" in (a.get("href") or "")), "")

        fighter_names = [a.get_text(" ", strip=True) for a in fighter_links]
        fighter_urls = [urljoin(BASE_URL, a.get("href", "")) for a in fighter_links]

        cells = [c.get_text(" ", strip=True) for c in row.select("td")]
        weight_class = cells[6] if len(cells) > 6 else ""
        method_hint = cells[7] if len(cells) > 7 else ""
        round_hint = cells[8] if len(cells) > 8 else ""
        time_hint = cells[9] if len(cells) > 9 else ""

        if len(fighter_names) >= 2:
            rows.append(
                {
                    "event_name": event_name,
                    "event_slug": event_slug,
                    "event_url": event_url,
                    "event_meta": meta_text,
                    "bout_url": bout_link,
                    "fighter_1_name": fighter_names[0],
                    "fighter_1_url": fighter_urls[0],
                    "fighter_2_name": fighter_names[1],
                    "fighter_2_url": fighter_urls[1],
                    "weight_class": weight_class,
                    "method_hint": method_hint,
                    "round_hint": round_hint,
                    "time_hint": time_hint,
                }
            )

    if not rows:
        raise RuntimeError("No fights found on event page. Check EVENT_URL.")

    return pd.DataFrame(rows), event_slug


def main() -> None:
    bucket = get_env("S3_BUCKET", "degenerative-investigator")
    region = get_env("AWS_REGION", "us-east-2")
    event_url = get_env("EVENT_URL", required=True)

    df, event_slug = parse_event_card(event_url)
    csv_key = f"raw/ufc/events/{event_slug}_card.csv"
    parquet_key = f"raw/ufc/events/parquets/{event_slug}_card.parquet"

    write_csv_and_parquet(df, bucket=bucket, csv_key=csv_key, parquet_key=parquet_key, region=region)
    print(f"Wrote {len(df)} fights to s3://{bucket}/{csv_key}")


if __name__ == "__main__":
    main()
