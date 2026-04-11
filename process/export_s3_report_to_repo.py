#!/usr/bin/env python3
import os
from pathlib import Path

import boto3


def main() -> None:
    bucket = os.environ.get("S3_BUCKET", "degenerative-investigator")
    region = os.environ.get("AWS_REGION", "us-east-2")
    report_key = os.environ.get(
        "REPORT_KEY",
        "processed/reports/ufc-327-prochazka-vs-ulberg_fight_report.md",
    )
    output_path = Path(os.environ.get("OUTPUT_PATH", "reports/latest_fight_report.md"))

    s3 = boto3.client("s3", region_name=region)
    obj = s3.get_object(Bucket=bucket, Key=report_key)
    text = obj["Body"].read().decode("utf-8-sig", errors="replace")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"Wrote {output_path} from s3://{bucket}/{report_key}")


if __name__ == "__main__":
    main()
