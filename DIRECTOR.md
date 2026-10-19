# degenerate_investigator Director Pack

## System

`degenerate_investigator` is a UFC-focused fight analytics pipeline that ingests event, fighter, odds, and news data, stores datasets in S3, builds matchup features, trains win-probability models, scores target events, and generates written fight-analysis reports.

## Existing capabilities

- UFC event and fighter ingestion.
- Historical fight/profile collection.
- Odds and news collection.
- Feature building and training-dataset generation.
- Win-probability model training.
- Event scoring and report generation.
- S3 storage helpers.
- GitHub Actions workflows for the main pipeline stages.

## Operating workflows

Main workflows are under `.github/workflows/`, including:

- `ufc_ingest_event.yml`
- `ufc_pull_fighter_profiles.yml`
- `ufc_pull_historical_fighter_profiles.yml`
- `ufc_pull_odds.yml`
- `ufc_pull_news.yml`
- `ufc_build_features.yml`
- `ufc_build_history.yml`
- `ufc_build_training_dataset.yml`
- `ufc_train_model.yml`
- `ufc_score_event.yml`
- `ufc_generate_report.yml`
- `export_latest_report_to_repo.yml`

## Readiness

Active development. Core pipeline stages exist, but live source availability, model quality, and workflow success should be verified before relying on current outputs.

## Authoritative state

- Human overview and configuration: `README.md`
- Implementation: `extract/`, `process/`, and `common/`
- Current generated report: `reports/latest_fight_report.md`
- Git/CI state: live GitHub repository and Actions runs
- AWS/S3 state: live AWS environment when inspection is available

If documentation and live behavior disagree, verify whether the documentation is stale or the implementation has changed before modifying architecture.

## Documentation pointers

- Human entry point: `README.md`
- Python dependencies: `requirements.txt`
- Workflow definitions: `.github/workflows/`
- Data extraction: `extract/`
- Processing/modelling/reporting: `process/`
- Shared helpers: `common/`
- Latest report: `reports/latest_fight_report.md`

## Do not rebuild

- Reuse existing extraction, processing, scoring, reporting, and S3 helpers before adding alternatives.
- Reuse existing GitHub Actions workflows where they already cover the required pipeline stage.
- Preserve existing AWS resource names during repository/account migration unless a separate migration explicitly requires changes.
- Do not add staking logic or bookmaker-targeted betting recommendations unless explicitly approved as a scope change.

## Operational warnings

- External data sources can change or fail; verify live workflow results before assuming a source is healthy.
- Never commit API keys, AWS credentials, or other secret values.
- Avoid infrastructure or service changes that introduce recurring cost without approval.
- This repository is private; documentation and portfolio publication are disabled in `project.yaml`.

## Standard deviations

None identified for Repository Standard V1.

## Continuation

- Outstanding work: continue validating source reliability, model quality, and end-to-end report generation.
- Next safe action: inspect the latest workflow runs and generated report before changing the pipeline design.
