# degenerate_investigator

Private UFC-focused fight data, modelling, scoring, and report-generation pipeline.

## Purpose

This repository:

- ingests event, fighter, odds, and news data;
- stores raw and processed datasets in S3;
- builds matchup features for modelling;
- trains win-probability models;
- scores a target event; and
- generates written fight-analysis reports.

It does **not** include staking logic or bookmaker-targeted betting recommendations.

## Status

Active development. The main pipeline stages and GitHub Actions workflows exist, but live source availability, model quality, and end-to-end report generation should be verified before relying on outputs.

## Architecture

```text
External fight / odds / news sources
            |
            v
        extract/
            |
            v
           S3
            |
            v
        process/
   features -> model -> score -> report
            |
            v
  reports/latest_fight_report.md
```

Shared S3/file helpers live under `common/`.

## Project layout

```text
.github/workflows/   Pipeline workflows
common/              Shared helpers
extract/             Event, fighter, odds, and news ingestion
process/             Features, training, scoring, and reporting
reports/             Generated report output
```

## AWS configuration

Required repository secrets include:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `OPENAI_API_KEY`
- `ODDS_API_KEY` (recommended)
- `NEWS_API_KEY` (optional)

Project defaults:

- bucket: `degenerative-investigator`
- region: `us-east-2`

Existing AWS resource names are intentionally preserved during repository/account migration.

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

## Main workflows

Workflows are under `.github/workflows/`. Key stages include:

- event/fighter ingestion;
- odds and news collection;
- feature and training-dataset generation;
- model training;
- event scoring;
- report generation; and
- exporting the latest report back to the repository.

## Running and validation

Install dependencies with:

```bash
pip install -r requirements.txt
```

Most operational runs are performed through the GitHub Actions workflows. Check the latest workflow results before assuming a source or pipeline stage is healthy.

There is currently no dedicated automated test suite in the repository.

## Documentation

- `DIRECTOR.md` — AI-agent operating instructions
- `project.yaml` — machine-readable project metadata
- `reports/latest_fight_report.md` — latest repository-exported report

## Roadmap

Near-term work is to keep validating data-source reliability, model quality, and end-to-end report generation before expanding the architecture or adding new product scope.
