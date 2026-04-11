#!/usr/bin/env python3
import time

import pandas as pd

from common.io_helpers import get_env, read_s3_csv, write_csv_and_parquet
from extract.ufc_fighter_profiles import parse_profile


def main() -> None:
    bucket = get_env("S3_BUCKET", "degenerative-investigator")
    region = get_env("AWS_REGION", "us-east-2")
    historical_fights_key = get_env("HISTORICAL_FIGHTS_KEY", "raw/ufc/fights/historical_fights.csv")
    delay = float(get_env("DELAY_BETWEEN_REQUESTS", "0.15"))
    test_rows = int(get_env("TEST_ROWS", "0"))

    df_fights = read_s3_csv(bucket=bucket, key=historical_fights_key, region=region)
    urls = sorted(set(df_fights["fighter_1_url"].dropna().tolist() + df_fights["fighter_2_url"].dropna().tolist()))
    if test_rows > 0:
        urls = urls[:test_rows]

    profiles = []
    for index, url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] {url}")
        try:
            profiles.append(parse_profile(url))
        except Exception as exc:
            profiles.append(
                {
                    "fighter_name": "",
                    "fighter_url": url,
                    "height": "",
                    "weight": "",
                    "reach": "",
                    "stance": "",
                    "dob": "",
                    "slpm": "",
                    "str_acc": "",
                    "sapm": "",
                    "str_def": "",
                    "td_avg": "",
                    "td_acc": "",
                    "td_def": "",
                    "sub_avg": "",
                    "career_wins_scraped": "",
                    "career_losses_scraped": "",
                    "career_other_scraped": "",
                    "recent_fights_scraped": "",
                    "profile_error": str(exc),
                }
            )
        time.sleep(delay)

    df = pd.DataFrame(profiles).drop_duplicates(subset=["fighter_url"])
    csv_key = "raw/ufc/fighters/historical_fighter_profiles.csv"
    parquet_key = "raw/ufc/fighters/parquets/historical_fighter_profiles.parquet"
    write_csv_and_parquet(df, bucket=bucket, csv_key=csv_key, parquet_key=parquet_key, region=region)
    print(f"Wrote {len(df)} historical fighter profiles to s3://{bucket}/{csv_key}")


if __name__ == "__main__":
    main()
