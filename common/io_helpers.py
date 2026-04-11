import io
import os
import re
from typing import Iterable, Optional

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError


DEFAULT_BUCKET = os.getenv("S3_BUCKET", "degenerative-investigator")
DEFAULT_REGION = os.getenv("AWS_REGION", "us-east-2")


def get_env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or str(value).strip() == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return str(value) if value is not None else ""


def slugify(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "unknown"


def s3_client(region: Optional[str] = None):
    return boto3.client("s3", region_name=region or DEFAULT_REGION)


def s3_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def read_s3_csv(bucket: str, key: str, region: Optional[str] = None) -> pd.DataFrame:
    s3 = s3_client(region)
    obj = s3.get_object(Bucket=bucket, Key=key)
    text = obj["Body"].read().decode("utf-8-sig", errors="replace")
    return pd.read_csv(io.StringIO(text))


def put_s3_text(bucket: str, key: str, text: str, content_type: str = "text/csv", region: Optional[str] = None) -> None:
    s3 = s3_client(region)
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8-sig"), ContentType=content_type)


def put_s3_bytes(bucket: str, key: str, payload: bytes, content_type: str, region: Optional[str] = None) -> None:
    s3 = s3_client(region)
    s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)


def write_csv_and_parquet(df: pd.DataFrame, bucket: str, csv_key: str, parquet_key: str, region: Optional[str] = None) -> None:
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    put_s3_text(bucket=bucket, key=csv_key, text=csv_buffer.getvalue(), content_type="text/csv", region=region)

    parquet_table = pa.Table.from_pandas(df, preserve_index=False)
    parquet_buffer = io.BytesIO()
    pq.write_table(parquet_table, parquet_buffer, compression="snappy")
    put_s3_bytes(
        bucket=bucket,
        key=parquet_key,
        payload=parquet_buffer.getvalue(),
        content_type="application/x-parquet",
        region=region,
    )


def pick_first(values: Iterable[str], default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default
