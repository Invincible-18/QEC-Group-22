# QEC data pipeline for AI (DSAIT4000, group 22)

A repeatable pipeline that turns the three supplied QEC archives into checked
Parquet tables, a PostgreSQL model and two ML tables, and then trains small
models on the ML tables. We use the course architecture:

```text
bronze (supplied zips, unchanged) -> silver (checked Parquet, one source each)
  -> gold (PostgreSQL) -> ml (two Parquet tables) -> Part II models
```

## Reports

- Part I design report:
  [starter/docs/part1-design-report.md](starter/docs/part1-design-report.md)
- Part II report: [starter/results/part2/report.md](starter/results/part2/report.md),
  written by the Part II command from the sections in `starter/docs/part2-report/`
  and the metrics of the same run
- The three SQL analyses: `starter/sql/q*.sql`, with their results and short
  interpretations in `starter/results/part1/analysis/`

## Setup

You need Docker (give it at least 8 GB of memory), `make` and a Bash terminal.
On Windows, use Ubuntu in WSL 2 with Docker Desktop's WSL integration turned on.

1. Copy the course's `datasets/` folder into this folder, so that
   `datasets/student-bundle/core/` exists. The course archives are not part of
   this submission.
2. Run `make bootstrap`. It creates `.env` from `.env.example`, builds the image,
   starts the services, copies the three archives into Bronze and checks the
   setup.

## Commands

Run these from this folder:

| What | Command |
|---|---|
| Part I | `docker compose exec workspace make run` |
| Part I twice, comparing all outputs | `docker compose exec workspace python -m quantum_lake_student.cli rerun-check` |
| Part II | `docker compose exec workspace make train` |
| Tests | `docker compose exec workspace make test` |

Part I takes about 40 seconds and Part II about 10. Part II reads only the two
ML tables, so run Part I first.

In a git checkout, `make run-part1`, `make rerun-check-part1` and
`make train-part2` run the same steps and also record the commit in `run.json`.
They refuse to run with uncommitted changes. The submitted results were made
this way.

`make down` stops the services and keeps the data. `make reset-platform`
deletes all data volumes.

## Where things are

| What | Where |
|---|---|
| Pipeline code | `starter/src/quantum_lake_student/stages/` |
| Gold tables with keys, constraints, indexes and checks | `stages/gold_google.py`, `gold_syndrome.py`, `gold_qasmbench.py`; the all-or-nothing load is in `stages/load_postgres.py` |
| SQL for the analyses and the ML export | `starter/sql/` (`q*.sql`, `ml_*.sql`) |
| Tests | `starter/tests/` |
| Bronze, Silver and ML tables | MinIO bucket `quantum-lake`: `bronze/`, `silver/<source>/<table>.parquet`, `ml/<table>.parquet` |
| Part I evidence | `starter/results/part1/`: issues, run record, row counts, source trace, trace examples, rerun check, analyses |
| Part II results | `starter/results/part2/`: predictions, metrics, run record, report, fitted models in `models/` |

To look at the data, open JupyterLab at
http://localhost:8888/lab?token=quantum-course, MinIO at http://localhost:9001
or Adminer (PostgreSQL, schema `gold`) at http://localhost:8080. The logins are
in `.env`.

`starter/results/part1/trace_examples.json` follows one syndrome and one Google
test example from the ML table through Gold and Silver to the Bronze files.
`starter/results/part2/trace_predictions.json` adds the predictions every model
made for those two examples.

Versions: Python 3.12.10 (`docker/pipeline.Dockerfile`), packages pinned in
`docker/requirements.txt`, random seed 2026. Part II records the versions it
used in `starter/results/part2/run.json`.

## Troubleshooting

- **Commands fail in PowerShell.** Use a Bash terminal; on Windows that is
  Ubuntu in WSL 2.
- **A port is already in use**, for example by a local PostgreSQL on 5432. Stop
  that program, or change the port in `.env` and run `make up`.
- **"Missing required Bronze object(s)", or `/course-data` is empty.** Check that
  `datasets/student-bundle/core/` exists, then run `make seed`. If the folder was
  added after the services started, also run
  `docker compose up -d --force-recreate workspace`.
- **`make run-part1` says "Uncommitted changes present".** Commit first, or use
  the container command from the table above; `run.json` then records the code
  revision as "unknown".
- **Files in `starter/results/` belong to root**, because the container runs as
  root. This can block `git switch`. Fix it with
  `docker compose exec workspace chown -R "$(id -u):$(id -g)" /workspace/results`.
- **JupyterLab shows a blank page.** The image was built with tornado 6.5.9,
  which breaks it. `docker/requirements.txt` pins 6.5.8, so rebuild with
  `make build` and then `make up`.

## AI usage

We used generative AI tools (Claude Sonnet, Claude Opus and Gemini Flash) as
assistants during the project:

- To discuss and check trade-offs in our design, such as evaluating Gold
  normalization (keeping decoders as separate entities versus wide columns),
  comparing detector storage approaches, and reviewing candidate keys;
- To draft initial boilerplate code, unit test outlines (like synthetic data
  fixtures) and SQL query skeletons;
- To help diagnose edge cases in source parsing, such as whitespace preceding
  semicolons in OpenQASM circuits; and
- To assist with structuring and editing drafts of the design reports.

All final design choices, relational schemas, pipeline transformations, test
assertions and report conclusions were decided, verified and tested by the team.
