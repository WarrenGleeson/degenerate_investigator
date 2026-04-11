#!/usr/bin/env python3
import json
from typing import Any

import pandas as pd
from openai import OpenAI

from common.io_helpers import get_env, put_s3_text, read_s3_csv, write_csv_and_parquet


PROMPT_TEMPLATE = """
You are writing a concise MMA fight preview for a structured analytics pipeline.
Return strict JSON with this schema:
{{
  "headline": "string",
  "summary": "string",
  "risk_note": "string"
}}

Write a neutral analysis for this matchup.
Rules:
- Do not mention betting advice, parlays, or staking.
- Mention where the model and market differ when relevant.
- Keep summary to 90-140 words.
- Be specific and readable.

Event: {event_name}
Weight class: {weight_class}
Fighter 1: {fighter_1_name}
Fighter 2: {fighter_2_name}
Predicted winner: {predicted_winner}
Fighter 1 win probability: {fighter_1_win_probability}
Fighter 2 win probability: {fighter_2_win_probability}
Confidence: {confidence_bucket}
Top signals: {top_signals}
Fighter 1 market implied probability: {fighter_1_market_implied_prob}
Fighter 2 market implied probability: {fighter_2_market_implied_prob}
Fighter 1 model-market delta: {fighter_1_model_market_delta}
Fighter 2 model-market delta: {fighter_2_model_market_delta}
Fighter 1 news summary: {fighter_1_news_summary}
Fighter 2 news summary: {fighter_2_news_summary}
""".strip()


def parse_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise RuntimeError("No JSON object found in model output.")
    return json.loads(text[start : end + 1])


def simple_fallback(row: pd.Series) -> dict[str, str]:
    winner = row.get("predicted_winner", "")
    signals = str(row.get("top_signals", "")).strip()
    conf = str(row.get("confidence_bucket", "low")).strip()
    delta1 = row.get("fighter_1_model_market_delta", "")
    delta2 = row.get("fighter_2_model_market_delta", "")
    summary = (
        f"The model leans toward {winner} in this {row.get('weight_class', '')} matchup. "
        f"Confidence is {conf}, with the strongest drivers listed as {signals or 'limited structured signals'}. "
        f"The market comparison shows fighter one delta {delta1} and fighter two delta {delta2}, "
        f"which helps flag where the model is more or less aggressive than current pricing."
    )
    return {
        "headline": f"{row.get('fighter_1_name', '')} vs {row.get('fighter_2_name', '')}",
        "summary": summary,
        "risk_note": "Use caution when confidence is low or when fighter context is sparse.",
    }


def load_news_summary(df_news: pd.DataFrame) -> pd.DataFrame:
    if df_news.empty:
        return pd.DataFrame(columns=["fighter_name_key", "news_summary"])
    df = df_news.copy()
    df["fighter_name_key"] = df["fighter_name"].astype(str).str.strip().str.lower()
    summary_df = (
        df.groupby("fighter_name_key", as_index=False)
        .agg(news_summary=("summary", "first"))
        .fillna("")
    )
    return summary_df


def deduplicate_matchups(df_pred: pd.DataFrame) -> pd.DataFrame:
    df = df_pred.copy().reset_index(drop=False).rename(columns={"index": "row_order"})
    df["fighter_1_key"] = df["fighter_1_name"].astype(str).str.strip().str.lower()
    df["fighter_2_key"] = df["fighter_2_name"].astype(str).str.strip().str.lower()
    df["matchup_key"] = df.apply(
        lambda row: "__vs__".join(sorted([row["fighter_1_key"], row["fighter_2_key"]])),
        axis=1,
    )

    conf_rank = {"high": 3, "medium": 2, "low": 1}
    df["confidence_rank"] = df.get("confidence_bucket", "").astype(str).str.lower().map(conf_rank).fillna(0)
    df["trained_model_first"] = (df.get("scoring_method", "").astype(str) == "trained_model").astype(int)

    chosen_rows = []
    for _, group in df.groupby("matchup_key", sort=False):
        group = group.sort_values(["trained_model_first", "confidence_rank", "row_order"], ascending=[False, False, True])
        chosen_rows.append(group.iloc[0])

    deduped = pd.DataFrame(chosen_rows).sort_values("row_order").reset_index(drop=True)
    return deduped


def build_event_overview(df: pd.DataFrame) -> list[str]:
    sections: list[str] = []
    if df.empty:
        return sections

    sections.append("## Event Overview\n")
    sections.append(f"- Fights covered: {len(df)}\n")
    main_event = df.iloc[0]
    sections.append(
        f"- Main event model lean: {main_event.get('predicted_winner', '')} over "
        f"{main_event.get('fighter_1_name', '') if main_event.get('predicted_winner', '') != main_event.get('fighter_1_name', '') else main_event.get('fighter_2_name', '')}"
        f" ({round(float(main_event.get('fighter_1_win_probability', 0.0) if main_event.get('predicted_winner', '') == main_event.get('fighter_1_name', '') else main_event.get('fighter_2_win_probability', 0.0)) * 100, 1)}%)\n"
    )
    high_conf = int((df.get("confidence_bucket", "") == "high").sum())
    sections.append(f"- High-confidence model leans: {high_conf}\n\n")

    sections.append("## Strongest Model Leans\n")
    strongest = df.copy()
    strongest["max_prob"] = strongest[["fighter_1_win_probability", "fighter_2_win_probability"]].max(axis=1)
    strongest = strongest.sort_values(["max_prob", "confidence_rank" if "confidence_rank" in strongest.columns else "event_name"], ascending=[False, False])
    for _, row in strongest.head(3).iterrows():
        sections.append(
            f"- {row.get('predicted_winner', '')} in {row.get('fighter_1_name', '')} vs {row.get('fighter_2_name', '')} "
            f"({round(float(max(row.get('fighter_1_win_probability', 0.0), row.get('fighter_2_win_probability', 0.0))) * 100, 1)}%, {row.get('confidence_bucket', '')})\n"
        )
    sections.append("\n")

    sections.append("## Largest Model-Market Gaps\n")
    gaps = []
    for _, row in df.iterrows():
        f1_gap = abs(float(row.get("fighter_1_model_market_delta", 0.0) or 0.0))
        f2_gap = abs(float(row.get("fighter_2_model_market_delta", 0.0) or 0.0))
        gaps.append((max(f1_gap, f2_gap), row))
    for _, row in sorted(gaps, key=lambda x: x[0], reverse=True)[:3]:
        sections.append(
            f"- {row.get('fighter_1_name', '')} vs {row.get('fighter_2_name', '')}: "
            f"largest gap on {row.get('predicted_winner', '')} view, confidence {row.get('confidence_bucket', '')}.\n"
        )
    sections.append("\n")
    return sections


def main() -> None:
    bucket = get_env("S3_BUCKET", "degenerative-investigator")
    region = get_env("AWS_REGION", "us-east-2")
    predictions_key = get_env("PREDICTIONS_KEY", required=True)
    news_key = get_env("NEWS_KEY", "raw/news/fighter_recent_news.csv")
    use_openai = get_env("USE_OPENAI_REPORTS", "true").strip().lower() == "true"

    df_pred = read_s3_csv(bucket=bucket, key=predictions_key, region=region)
    df_pred = deduplicate_matchups(df_pred)
    try:
        df_news = read_s3_csv(bucket=bucket, key=news_key, region=region)
    except Exception:
        df_news = pd.DataFrame()

    news_summary = load_news_summary(df_news)
    f1_news = news_summary.rename(columns={"fighter_name_key": "fighter_1_key", "news_summary": "fighter_1_news_summary"})
    f2_news = news_summary.rename(columns={"fighter_name_key": "fighter_2_key", "news_summary": "fighter_2_news_summary"})
    df = df_pred.merge(f1_news, on="fighter_1_key", how="left").merge(f2_news, on="fighter_2_key", how="left")
    df = df.fillna("")

    client = OpenAI(api_key=get_env("OPENAI_API_KEY", required=False)) if use_openai else None

    report_rows = []
    markdown_sections = []
    event_name = str(df["event_name"].iloc[0]) if not df.empty else "Event Report"
    markdown_sections.append(f"# {event_name} Report\n\n")
    markdown_sections.extend(build_event_overview(df))

    prompt_fields = [
        "event_name",
        "weight_class",
        "fighter_1_name",
        "fighter_2_name",
        "predicted_winner",
        "fighter_1_win_probability",
        "fighter_2_win_probability",
        "confidence_bucket",
        "top_signals",
        "fighter_1_market_implied_prob",
        "fighter_2_market_implied_prob",
        "fighter_1_model_market_delta",
        "fighter_2_model_market_delta",
        "fighter_1_news_summary",
        "fighter_2_news_summary",
    ]

    for _, row in df.iterrows():
        payload = None
        if client is not None:
            try:
                response = client.responses.create(
                    model=get_env("OPENAI_MODEL", "gpt-4.1-mini"),
                    input=PROMPT_TEMPLATE.format(**{k: row.get(k, "") for k in prompt_fields}),
                    max_output_tokens=500,
                )
                payload = parse_json(response.output_text)
            except Exception:
                payload = None

        if payload is None:
            payload = simple_fallback(row)

        report_rows.append(
            {
                "event_name": row.get("event_name", ""),
                "fighter_1_name": row.get("fighter_1_name", ""),
                "fighter_2_name": row.get("fighter_2_name", ""),
                "predicted_winner": row.get("predicted_winner", ""),
                "fighter_1_win_probability": row.get("fighter_1_win_probability", ""),
                "fighter_2_win_probability": row.get("fighter_2_win_probability", ""),
                "confidence_bucket": row.get("confidence_bucket", ""),
                "headline": payload.get("headline", ""),
                "summary": payload.get("summary", ""),
                "risk_note": payload.get("risk_note", ""),
                "row_order": row.get("row_order", ""),
            }
        )

        markdown_sections.append(f"## {payload.get('headline', row.get('fighter_1_name', '') + ' vs ' + row.get('fighter_2_name', ''))}\n")
        markdown_sections.append(f"- Predicted winner: {row.get('predicted_winner', '')}\n")
        markdown_sections.append(f"- Confidence: {row.get('confidence_bucket', '')}\n")
        markdown_sections.append(f"- Fighter 1 win probability: {row.get('fighter_1_win_probability', '')}\n")
        markdown_sections.append(f"- Fighter 2 win probability: {row.get('fighter_2_win_probability', '')}\n\n")
        markdown_sections.append(payload.get("summary", "") + "\n\n")
        markdown_sections.append("Risk note: " + payload.get("risk_note", "") + "\n\n")

    df_report = pd.DataFrame(report_rows).sort_values("row_order").drop(columns=["row_order"], errors="ignore")
    event_slug = str(df["event_slug"].iloc[0]) if not df.empty else "event"
    csv_key = f"processed/reports/{event_slug}_fight_report.csv"
    parquet_key = f"processed/reports/parquets/{event_slug}_fight_report.parquet"
    md_key = f"processed/reports/{event_slug}_fight_report.md"

    write_csv_and_parquet(df_report, bucket=bucket, csv_key=csv_key, parquet_key=parquet_key, region=region)
    put_s3_text(bucket=bucket, key=md_key, text="".join(markdown_sections), content_type="text/markdown", region=region)
    print(f"Wrote {len(df_report)} report rows to s3://{bucket}/{csv_key}")
    print(f"Wrote markdown report to s3://{bucket}/{md_key}")


if __name__ == "__main__":
    main()
