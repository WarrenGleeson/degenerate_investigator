#!/usr/bin/env python3
import io
import json
import time
from typing import Any

import pandas as pd
from openai import OpenAI

from common.io_helpers import get_env, read_s3_csv, write_csv_and_parquet


PROMPT_TEMPLATE = """
Find recent factual reporting for the fighter below.
Return strict JSON with this schema:
{
  "fighter_name": string,
  "summary": string,
  "items": [
    {
      "title": string,
      "source": string,
      "published_date": string,
      "url": string,
      "label": string,
      "note": string
    }
  ]
}

Focus on recent injuries, weight-cut issues, training-camp changes, layoffs, short-notice status, legal/personal disruptions, or coach comments.
Fighter: {fighter_name}
""".strip()


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError("Model did not return JSON.")
    return json.loads(text[start : end + 1])


def main() -> None:
    bucket = get_env("S3_BUCKET", "degenerative-investigator")
    region = get_env("AWS_REGION", "us-east-2")
    profiles_key = get_env("FIGHTER_PROFILES_KEY", "raw/ufc/fighters/fighter_profiles.csv")
    test_rows = int(get_env("TEST_ROWS", "0"))
    delay = float(get_env("DELAY_BETWEEN_REQUESTS", "0.5"))

    client = OpenAI(api_key=get_env("OPENAI_API_KEY", required=True))
    df_profiles = read_s3_csv(bucket=bucket, key=profiles_key, region=region)
    fighter_names = df_profiles["fighter_name"].dropna().astype(str).str.strip().tolist()
    if test_rows > 0:
        fighter_names = fighter_names[:test_rows]

    rows = []
    for index, fighter_name in enumerate(fighter_names, start=1):
        print(f"[{index}/{len(fighter_names)}] {fighter_name}")
        response = client.responses.create(
            model=get_env("OPENAI_MODEL", "gpt-4.1-mini"),
            input=PROMPT_TEMPLATE.format(fighter_name=fighter_name),
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            max_output_tokens=900,
        )
        payload = extract_json(response.output_text)
        summary = str(payload.get("summary", "")).strip()
        for item in payload.get("items", []):
            rows.append(
                {
                    "fighter_name": fighter_name,
                    "summary": summary,
                    "title": item.get("title", ""),
                    "source": item.get("source", ""),
                    "published_date": item.get("published_date", ""),
                    "url": item.get("url", ""),
                    "label": item.get("label", ""),
                    "note": item.get("note", ""),
                }
            )
        time.sleep(delay)

    df = pd.DataFrame(rows)
    csv_key = "raw/news/fighter_recent_news.csv"
    parquet_key = "raw/news/parquets/fighter_recent_news.parquet"
    write_csv_and_parquet(df, bucket=bucket, csv_key=csv_key, parquet_key=parquet_key, region=region)
    print(f"Wrote {len(df)} news rows to s3://{bucket}/{csv_key}")


if __name__ == "__main__":
    main()
