#!/usr/bin/env python3
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from common.io_helpers import get_env, write_csv_and_parquet


BASE_URL = "http://ufcstats.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_soup(url: str) -> BeautifulSoup:
    response = requests.get(url, timeout=30, headers=HEADERS)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def completed_event_urls(max_events: int) -> list[str]:
    soup = fetch_soup(f"{BASE_URL}/statistics/events/completed?page=all")
    urls = []
    for link in soup.select("a[href*='event-details']"):
        href = link.get("href", "")
        if href and href not in urls:
            urls.append(urljoin(BASE_URL, href))
    return urls[:max_events]


def parse_event(event_url: str) -> list[dict]:
    soup = fetch_soup(event_url)
    title_node = soup.select_one(".b-content__title-highlight") or soup.select_one(".b-content__title")
    event_name = title_node.get_text(" ", strip=True) if title_node else event_url

    rows = []
    for row in soup.select("tr.b-fight-details__table-row"):
        links = row.select("a")
        fighter_links = [a for a in links if "/fighter-details/" in (a.get("href") or "")]
        fighter_names = [a.get_text(" ", strip=True) for a in fighter_links]
        fighter_urls = [urljoin(BASE_URL, a.get("href", "")) for a in fighter_links]
        cells = [c.get_text(" ", strip=True) for c in row.select("td")]
        result = cells[0] if len(cells) > 0 else ""
        if len(fighter_names) >= 2:
            rows.append(
                {
                    "event_name": event_name,
                    "event_url": event_url,
                    "fighter_1_name": fighter_names[0],
                    "fighter_1_url": fighter_urls[0],
                    "fighter_2_name": fighter_names[1],
                    "fighter_2_url": fighter_urls[1],
                    "weight_class": cells[6] if len(cells) > 6 else "",
                    "fighter_1_win": 1 if result.upper() == "W" else 0,
                }
            )
    return rows


def main() -> None:
    bucket = get_env("S3_BUCKET", "degenerative-investigator")
    region = get_env("AWS_REGION", "us-east-2")
    max_events = int(get_env("MAX_EVENTS", "150"))
    delay = float(get_env("DELAY_BETWEEN_REQUESTS", "0.2"))

    rows = []
    event_urls = completed_event_urls(max_events=max_events)
    for index, event_url in enumerate(event_urls, start=1):
        print(f"[{index}/{len(event_urls)}] {event_url}")
        rows.extend(parse_event(event_url))
        time.sleep(delay)

    df = pd.DataFrame(rows)
    csv_key = "raw/ufc/fights/historical_fights.csv"
    parquet_key = "raw/ufc/fights/parquets/historical_fights.parquet"
    write_csv_and_parquet(df, bucket=bucket, csv_key=csv_key, parquet_key=parquet_key, region=region)
    print(f"Wrote {len(df)} historical fights to s3://{bucket}/{csv_key}")


if __name__ == "__main__":
    main()
