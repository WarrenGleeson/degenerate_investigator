# degenerate_investigator

Fight data pipelines, event analysis, and outcome modelling.

## Purpose

This repo is for building a UFC-first fight analytics pipeline that:
- ingests event, fighter, odds, and news data
- stores raw and processed datasets in S3
- builds matchup features for modeling
- trains win-probability models
- scores a target event
- generates written fight-analysis reports

This repo does **not** include staking logic or bookmaker-targeted betting recommendations.

## Project layout

```text
.github/workflows/
common/
extract/
process/
prompts/
tasks/
```

## AWS configuration

Set these repository secrets in GitHub:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `OPENAI_API_KEY`
- `ODDS_API_KEY` (recommended)
- `NEWS_API_KEY` (optional)

Project defaults:
- bucket: `degenerative-investigator`
- region: `us-east-2`

## S3 layout

```text
raw/ufc/events/
raw/ufc/fighters/
raw/ufc/fights/
raw/odds/
raw/news/
processed/ufc/
processed/ufc/parquets/
processed/reports/
```

## MVP workflows

- `ufc_ingest_event.yml`
- `ufc_pull_odds.yml`
- `ufc_pull_news.yml`
- `ufc_build_features.yml`
- `ufc_train_model.yml`
- `ufc_score_event.yml`

## Tonight's fastest path

1. Add GitHub secrets.
2. Run `ufc_ingest_event.yml` with a UFC Stats event URL.
3. Run `ufc_pull_odds.yml`.
4. Run `ufc_pull_news.yml`.
5. Run `ufc_build_features.yml`.
6. Run `ufc_train_model.yml` once historical data is available.
7. Run `ufc_score_event.yml`.

## Notes

- UFC Stats is the primary free structured source for event, fighter, and fight-level information.
- The Odds API is the primary current-odds source for the MVP.
- OpenAI web search is used for recent fighter context and report generation.
