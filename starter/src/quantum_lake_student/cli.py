"""Command-line entry point for the student workspace."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .config import Settings
from .connections import bronze_inventory, check_platform
from .stages.prepare_data import run as run_prepare_data


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


def command_run(settings: Settings) -> int:
    archive_candidates = [
        Path("/course-data/raw/source=qec_syndromes/syndromes_dataset.zip"),
        Path("../datasets/student-bundle/core/raw/source=qec_syndromes/syndromes_dataset.zip"),
    ]
    archive = next((candidate for candidate in archive_candidates if candidate.exists()), None)
    if archive is None:
        console.print("[red]Syndrome archive was not found.[/red]")
        return 2
    run_id = datetime.now(UTC).strftime("syndrome-%Y%m%dT%H%M%SZ")
    result = run_prepare_data(run_id, archive, settings.local_lake_root)
    console.print(
        f"[green]OK[/green] syndrome Silver: {result.output_count:,} rows, "
        f"{result.issue_count:,} issues written under {settings.local_lake_root}"
    )
    return 0


def command_train(_: Settings) -> int:
    console.print(
        "[yellow]The AI/ML stage is intentionally unimplemented.[/yellow]\n"
        "Consume the required ML input tables through the supplied helpers and "
        "write model files and the required results/part2 files."
    )
    return 2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=("check", "inventory", "run", "train"),
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
        "train": command_train,
    }
    raise SystemExit(commands[arguments.command](settings))


if __name__ == "__main__":
    main()
