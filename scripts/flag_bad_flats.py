#!/usr/bin/env python3
"""Move STACKED_FLATS session folders for bad-weather nights to BAD_STACKED_FLATS.

A night is considered "bad weather" when its % active (imaging time as a
fraction of the astronomical dark window) falls below --threshold (default 80).

The flat folder naming convention DATE_YYYY-MM-DD uses the same "evening date"
as night_report.csv: frames after midnight but before noon local time belong to
the previous calendar date, so STACKED_FLATS/DATE_2025-11-27 maps directly to
the Date=2025-11-27 row in the report.

Run with --dry-run (default) to preview; add --execute to perform the moves.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from tabulate import tabulate

from report_paths import NIGHT_REPORT_CSV

DEFAULT_NINA_ROOT = Path("/Volumes/FileStore/ASTRO/AARO/NINA")
DEFAULT_THRESHOLD = 80.0

FOLDER_DATE_RE = re.compile(r"^DATE_(?P<date>\d{4}-\d{2}-\d{2})$")

# ── CSV loading ────────────────────────────────────────────────────────────────

def load_report(report_path: Path) -> dict[date, float]:
    """Return mapping of night date → % active from the night report CSV."""
    df = pd.read_csv(report_path)
    df.columns = df.columns.str.strip()
    if "% active" not in df.columns or "Date" not in df.columns:
        raise SystemExit(
            f"Expected columns 'Date' and '% active' in {report_path}.\n"
            f"Found: {list(df.columns)}"
        )
    result: dict[date, float] = {}
    for _, row in df.iterrows():
        try:
            d = date.fromisoformat(str(row["Date"]).strip())
            result[d] = float(row["% active"])
        except (ValueError, TypeError):
            continue
    return result


# ── Flat-folder helpers ────────────────────────────────────────────────────────

def flat_folders(stacked_dir: Path) -> list[tuple[date, Path]]:
    """List all DATE_YYYY-MM-DD subdirectories sorted by date."""
    found: list[tuple[date, Path]] = []
    if not stacked_dir.is_dir():
        return found
    for entry in sorted(stacked_dir.iterdir()):
        m = FOLDER_DATE_RE.match(entry.name)
        if m and entry.is_dir():
            found.append((date.fromisoformat(m.group("date")), entry))
    return found


def dest_dir_for(src: Path, bad_root: Path) -> Path:
    return bad_root / src.name


# ── Core logic ────────────────────────────────────────────────────────────────

def evaluate_dirs(
    stacked_root: Path,
    bad_root: Path,
    activity: dict[date, float],
    threshold: float,
) -> list[dict]:
    rows = []
    for night_date, folder in flat_folders(stacked_root):
        pct = activity.get(night_date)
        dest = dest_dir_for(folder, bad_root)
        if pct is None:
            status = "SKIP"
            note = "no report entry"
        elif pct < threshold:
            status = "MOVE"
            note = f"% active = {pct:.1f} < {threshold:g}"
        else:
            status = "KEEP"
            note = f"% active = {pct:.1f}"
        rows.append(
            {
                "date": night_date,
                "folder": folder,
                "dest": dest,
                "pct_active": pct,
                "status": status,
                "note": note,
            }
        )
    return rows


def execute_moves(rows: list[dict], bad_root: Path, dry_run: bool) -> tuple[int, int, int]:
    moved = kept = skipped = 0
    if not dry_run:
        bad_root.mkdir(parents=True, exist_ok=True)

    for row in rows:
        if row["status"] == "MOVE":
            if dry_run:
                moved += 1
            else:
                dest = row["dest"]
                if dest.exists():
                    print(
                        f"  WARNING: destination already exists, skipping: {dest}",
                        file=sys.stderr,
                    )
                    skipped += 1
                else:
                    shutil.move(str(row["folder"]), str(dest))
                    moved += 1
        elif row["status"] == "KEEP":
            kept += 1
        else:
            skipped += 1

    return moved, kept, skipped


# ── Display ───────────────────────────────────────────────────────────────────

def print_table(rows: list[dict], label: str, dry_run: bool) -> None:
    if not rows:
        print(f"\n{label}: no DATE_* folders found.")
        return

    mode = "[DRY RUN] " if dry_run else ""
    print(f"\n{mode}{label}")
    table = [
        [
            str(r["date"]),
            f"{r['pct_active']:.1f}" if r["pct_active"] is not None else "—",
            r["status"],
            r["note"],
        ]
        for r in rows
    ]
    print(
        tabulate(
            table,
            headers=["Date", "% active", "Action", "Note"],
            tablefmt="simple",
        )
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Move bad-weather STACKED_FLATS session folders to BAD_STACKED_FLATS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Preview what would be moved (default)\n"
            "  flag_bad_flats.py --dry-run\n\n"
            "  # Actually move the folders\n"
            "  flag_bad_flats.py --execute\n\n"
            "  # Custom threshold\n"
            "  flag_bad_flats.py --threshold 70 --execute\n"
        ),
    )
    parser.add_argument(
        "--nina-root",
        type=Path,
        default=DEFAULT_NINA_ROOT,
        help=f"Path to NINA root directory (default: {DEFAULT_NINA_ROOT})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=NIGHT_REPORT_CSV,
        help=f"Path to night_report.csv (default: {NIGHT_REPORT_CSV})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            f"% active cutoff — nights below this are flagged as bad weather "
            f"(default: {DEFAULT_THRESHOLD})"
        ),
    )
    parser.add_argument(
        "--include-split",
        action="store_true",
        help="Also process STACKED_FLATS_SPLIT → BAD_STACKED_FLATS_SPLIT",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=True,
        help="Preview moves without touching the filesystem (default)",
    )
    mode_group.add_argument(
        "--execute",
        dest="dry_run",
        action="store_false",
        help="Actually move the folders",
    )

    args = parser.parse_args()

    if not args.nina_root.is_dir():
        raise SystemExit(f"NINA root not found: {args.nina_root}")
    if not args.report.is_file():
        raise SystemExit(f"Night report not found: {args.report}")

    activity = load_report(args.report)

    targets = [
        (
            args.nina_root / "STACKED_FLATS",
            args.nina_root / "BAD_STACKED_FLATS",
            "STACKED_FLATS",
        )
    ]
    if args.include_split:
        targets.append(
            (
                args.nina_root / "STACKED_FLATS_SPLIT",
                args.nina_root / "BAD_STACKED_FLATS_SPLIT",
                "STACKED_FLATS_SPLIT",
            )
        )

    total_moved = total_kept = total_skipped = 0

    for stacked_root, bad_root, label in targets:
        rows = evaluate_dirs(stacked_root, bad_root, activity, args.threshold)
        print_table(rows, label, args.dry_run)
        moved, kept, skipped = execute_moves(rows, bad_root, args.dry_run)
        total_moved += moved
        total_kept += kept
        total_skipped += skipped

    mode_note = " (dry run — use --execute to apply)" if args.dry_run else ""
    print(
        f"\nSummary{mode_note}: "
        f"{total_moved} to move, {total_kept} to keep, {total_skipped} skipped"
    )


if __name__ == "__main__":
    main()
