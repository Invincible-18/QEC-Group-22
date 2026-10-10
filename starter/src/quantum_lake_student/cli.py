"""Command-line entry point for the student workspace."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from rich.console import Console
from rich.table import Table

from .config import Settings
from .connections import bronze_inventory, check_platform
from .stages.build_ml_tables import run as run_build_ml_tables
from .stages.load_postgres import run as run_load_postgres
from .stages.prepare_data import run as run_prepare_data
from .stages.register_sources import run as run_register_sources


console = Console()


def command_check(settings: Settings) -> int:
    result = check_platform(settings)
    for service, message in result.items():
        console.print(f"[green]OK[/green] {service}: {message}")
    return 0


def command_inventory(settings: Settings) -> int:
    table = Table(title="Supplied course data files (kept unchanged)")
    table.add_column("Stored path")
    table.add_column("Bytes", justify="right")
    for key, size in bronze_inventory(settings):
        table.add_row(key, f"{size:,}")
    console.print(table)
    return 0


def command_run(_: Settings) -> int:
    run_id = datetime.now(UTC).strftime("part1-%Y%m%dT%H%M%SZ")
    bronze = run_register_sources(run_id)
    console.print(
        f"[green]OK[/green] register_sources: {bronze.output_count} Bronze objects verified "
        "against the release manifest"
    )
    result = run_prepare_data(run_id)
    console.print(
        f"[green]OK[/green] prepare_data: {result.output_count:,} Silver rows written, "
        f"{result.issue_count:,} issues"
    )
    gold = run_load_postgres(run_id)
    console.print(f"[green]OK[/green] load_postgres: {gold.output_count:,} Gold rows loaded")
    ml = run_build_ml_tables(run_id)
    console.print(
        f"[green]OK[/green] build_ml_tables: analyses in results/part1/analysis, "
        f"{ml.output_count:,} rows across ml_google_decoder_example and ml_syndrome_decoder_example"
    )
    return 0


def command_train(_: Settings) -> int:
    from .stages.train import run as run_train

    model_run_id = datetime.now(UTC).strftime("part2-%Y%m%dT%H%M%SZ")
    result = run_train(model_run_id)
    console.print(
        f"[green]OK[/green] train: {result.output_count} models evaluated from "
        f"{result.input_count:,} ML rows; results in results/part2"
    )
    return 0


def command_rerun_check(_: Settings) -> int:
    from .rerun_check import main as rerun_check_main

    return rerun_check_main()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=("check", "inventory", "run", "rerun-check", "train"),
        help="Action to perform",
    )
    return result


def main() -> None:
    arguments = parser().parse_args()
    settings = Settings.from_environment()
    commands = {
        "check": command_check,
        "inventory": command_inventory,
        "run": command_run,
        "rerun-check": command_rerun_check,
        "train": command_train,
    }
    raise SystemExit(commands[arguments.command](settings))


if __name__ == "__main__":
    main()
