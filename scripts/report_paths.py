"""Canonical paths for pipeline reports and plots."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_ROOT = PROJECT_ROOT / "output" / "night-activity"
DATA_DIR = REPORT_ROOT / "data"
PLOTS_DIR = REPORT_ROOT / "plots"
OVERVIEW_DIR = PLOTS_DIR / "overview"
MONTHLY_DIR = PLOTS_DIR / "by-month"
TARGET_PLOTS_ROOT = PROJECT_ROOT / "output" / "target-plots"

NIGHT_REPORT_CSV = DATA_DIR / "night_report.csv"
OVERVIEW_PLOT = OVERVIEW_DIR / "all-nights-longitudinal.png"


def target_slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "target"


def monthly_plot_path(period: pd.Period) -> Path:
    """e.g. output/night-activity/plots/by-month/2025/2025-11-november-activity.png"""
    year = period.strftime("%Y")
    month_num = period.strftime("%m")
    month_name = period.strftime("%B").lower()
    return MONTHLY_DIR / year / f"{year}-{month_num}-{month_name}-activity.png"


def target_fwhm_plot_path(target_name: str) -> Path:
    """e.g. output/target-plots/m81-and-m82/per-sub-fwhm-by-session.png"""
    return TARGET_PLOTS_ROOT / target_slug(target_name) / "per-sub-fwhm-by-session.png"


def ensure_report_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)


def ensure_target_plot_dir(target_name: str) -> Path:
    path = TARGET_PLOTS_ROOT / target_slug(target_name)
    path.mkdir(parents=True, exist_ok=True)
    return path
