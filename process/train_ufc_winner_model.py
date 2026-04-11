#!/usr/bin/env python3
import json
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common.io_helpers import get_env, read_s3_csv, put_s3_bytes, put_s3_text, write_csv_and_parquet


FEATURE_COLS = [
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


def main() -> None:
    bucket = get_env("S3_BUCKET", "degenerative-investigator")
    region = get_env("AWS_REGION", "us-east-2")
    training_key = get_env("TRAINING_DATA_KEY", required=True)

    df = read_s3_csv(bucket=bucket, key=training_key, region=region)
    if "fighter_1_win" not in df.columns:
        raise RuntimeError("Training dataset must include fighter_1_win as the target column.")

    X = df[FEATURE_COLS].copy()
    y = pd.to_numeric(df["fighter_1_win"], errors="coerce")

    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            (
                "pre",
                ColumnTransformer(
                    transformers=[("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), FEATURE_COLS)]
                ),
            ),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42, min_samples_leaf=3)),
        ]
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "log_loss": float(log_loss(y_test, proba)),
        "features": FEATURE_COLS,
    }

    put_s3_text(bucket=bucket, key="processed/ufc/model_metrics.json", text=json.dumps(metrics, indent=2), content_type="application/json", region=region)
    put_s3_bytes(bucket=bucket, key="processed/ufc/model_artifacts.pkl", payload=pickle.dumps(model), content_type="application/octet-stream", region=region)

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": model.named_steps["clf"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    write_csv_and_parquet(
        feature_importance,
        bucket=bucket,
        csv_key="processed/ufc/feature_importance.csv",
        parquet_key="processed/ufc/parquets/feature_importance.parquet",
        region=region,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
