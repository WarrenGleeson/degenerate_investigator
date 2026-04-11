#!/usr/bin/env python3
import json
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common.io_helpers import get_env, put_s3_bytes, put_s3_text, read_s3_csv, write_csv_and_parquet


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


def build_pipeline(feature_cols: list[str], use_dummy: bool = False) -> Pipeline:
    clf = DummyClassifier(strategy="prior") if use_dummy else RandomForestClassifier(n_estimators=300, random_state=42, min_samples_leaf=3)
    return Pipeline(
        steps=[
            (
                "pre",
                ColumnTransformer(
                    transformers=[("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), feature_cols)]
                ),
            ),
            ("clf", clf),
        ]
    )


def main() -> None:
    bucket = get_env("S3_BUCKET", "degenerative-investigator")
    region = get_env("AWS_REGION", "us-east-2")
    training_key = get_env("TRAINING_DATA_KEY", required=True)

    df = read_s3_csv(bucket=bucket, key=training_key, region=region)
    if "fighter_1_win" not in df.columns:
        raise RuntimeError("Training dataset must include fighter_1_win as the target column.")

    available_feature_cols = [col for col in FEATURE_COLS if col in df.columns and df[col].notna().any()]
    if not available_feature_cols:
        raise RuntimeError("No usable feature columns found in the training dataset.")

    X = df[available_feature_cols].copy()
    y = pd.to_numeric(df["fighter_1_win"], errors="coerce")

    valid_mask = y.notna()
    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    unique_classes = sorted(y.unique().tolist())
    use_dummy = len(unique_classes) < 2

    if len(X) < 10:
        raise RuntimeError("Training dataset is too small to train a model.")

    if use_dummy:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

    model = build_pipeline(available_feature_cols, use_dummy=use_dummy)
    model.fit(X_train, y_train)

    proba_raw = model.predict_proba(X_test)
    if proba_raw.shape[1] == 1:
        only_class = int(model.named_steps["clf"].classes_[0])
        proba = pd.Series(1.0 if only_class == 1 else 0.0, index=X_test.index).values
    else:
        class_index = list(model.named_steps["clf"].classes_).index(1)
        proba = proba_raw[:, class_index]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "log_loss": float(log_loss(y_test, proba, labels=[0, 1])),
        "features": available_feature_cols,
        "unique_classes": unique_classes,
        "model_type": "dummy_prior" if use_dummy else "random_forest",
    }
    if len(unique_classes) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))

    artifact = {
        "model": model,
        "feature_cols": available_feature_cols,
    }

    put_s3_text(bucket=bucket, key="processed/ufc/model_metrics.json", text=json.dumps(metrics, indent=2), content_type="application/json", region=region)
    put_s3_bytes(bucket=bucket, key="processed/ufc/model_artifacts.pkl", payload=pickle.dumps(artifact), content_type="application/octet-stream", region=region)

    if not use_dummy:
        feature_importance = pd.DataFrame(
            {
                "feature": available_feature_cols,
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
