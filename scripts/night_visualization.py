#!/usr/bin/env python3
"""Longitudinal plotnine charts of observing-night metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from plotnine import (
    aes,
    element_text,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_x_datetime,
    scale_y_continuous,
    theme,
    theme_minimal,
)

from report_paths import (
    NIGHT_REPORT_CSV,
    OVERVIEW_PLOT,
    ensure_report_dirs,
    monthly_plot_path,
)

LEGEND_HANDLES = [
    Line2D([0], [0], color="#2563eb", linewidth=1.15, marker="o", label="% active"),
    Line2D([0], [0], color="#94a3b8", alpha=0.35, linewidth=8, label="Dark window (hrs)"),
    Line2D([0], [0], color="#d97706", linewidth=1.15, marker="o", label="Avg FWHM"),
    Line2D([0], [0], color="#dc2626", linewidth=1.15, marker="s", label="Closure time (hrs)"),
    Line2D([0], [0], color="#7c3aed", linewidth=1.15, marker="^", label="Closure count"),
]


def load_report(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["Date"])
    df["month"] = df["date"].dt.to_period("M")
    return df.sort_values("date").reset_index(drop=True)


def month_label(period: pd.Period) -> str:
    return period.strftime("%B %Y")


def build_plot(
    df: pd.DataFrame,
    *,
    title: str,
    x_limits: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    x_breaks: str = "1 month",
    x_labels: str = "%b %Y",
    figure_size: tuple[float, float] = (14, 6),
) -> ggplot:
    x_scale = scale_x_datetime(date_breaks=x_breaks, date_labels=x_labels)
    if x_limits is not None:
        x_scale = scale_x_datetime(
            limits=x_limits,
            date_breaks=x_breaks,
            date_labels=x_labels,
        )

    return (
        ggplot(df, aes(x="date"))
        + geom_line(aes(y="% active"), color="#2563eb", size=1.15)
        + geom_point(aes(y="% active"), color="#2563eb", size=2.0)
        + x_scale
        + scale_y_continuous(name="Left: % active", limits=(0, 105))
        + labs(
            title=title,
            subtitle="Background ribbon = astronomical dark window (hrs); numeric metrics on right axis",
            x="Observing night (evening date)",
        )
        + theme_minimal(base_size=11)
        + theme(
            plot_title=element_text(weight="bold"),
            figure_size=figure_size,
        )
    )


def add_overlays(fig: plt.Figure, df: pd.DataFrame, *, marker_size: float = 3.5) -> None:
    ax = fig.axes[0]
    ax2 = ax.twinx()

    dates = mdates.date2num(df["date"])
    dark = df["Dark window (hrs)"].to_numpy()
    fwhm = df["Avg FWHM"].to_numpy()
    closure_hrs = df["Total closure time (hrs)"].to_numpy()
    closure_count = df["Roof closure count"].to_numpy()

    gap_days = df["date"].diff().dt.days.fillna(0)
    segment_starts = [0] + [i for i in range(1, len(df)) if gap_days.iloc[i] > 2]
    segment_starts.append(len(df))
    for start, end in zip(segment_starts, segment_starts[1:]):
        ax2.fill_between(
            dates[start:end],
            0,
            dark[start:end],
            color="#94a3b8",
            alpha=0.18,
            zorder=1,
        )

    ax2.plot(dates, fwhm, color="#d97706", linewidth=1.15, marker="o", markersize=marker_size, zorder=4)
    ax2.plot(dates, closure_hrs, color="#dc2626", linewidth=1.15, marker="s", markersize=marker_size, zorder=4)
    ax2.plot(dates, closure_count, color="#7c3aed", linewidth=1.15, marker="^", markersize=marker_size, zorder=4)

    ax2.set_ylabel("Right: dark window · Avg FWHM · closure count · closure time (hrs)", color="#555555")
    ax2.tick_params(axis="y", colors="#555555")
    right_max = max(dark.max(), fwhm.max(), closure_hrs.max(), closure_count.max(), 1)
    ax2.set_ylim(0, right_max * 1.15)
    ax.legend(handles=LEGEND_HANDLES, loc="upper left", framealpha=0.92, fontsize=8)


def save_plot(p: ggplot, df: pd.DataFrame, output_path: Path, *, marker_size: float = 3.5) -> None:
    fig = p.draw()
    add_overlays(fig, df, marker_size=marker_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_overview(df: pd.DataFrame, output_path: Path) -> None:
    p = build_plot(
        df,
        title="AARO Observing Nights — Activity, Seeing & Roof Closures",
        x_breaks="1 month",
        x_labels="%b %Y",
        figure_size=(14, 6),
    )
    save_plot(p, df, output_path)


def render_month(df: pd.DataFrame, period: pd.Period, output_path: Path) -> None:
    month_df = df[df["month"] == period].reset_index(drop=True)
    if month_df.empty:
        return

    month_start = period.to_timestamp()
    month_end = (period + 1).to_timestamp() - pd.Timedelta(days=1)
    p = build_plot(
        month_df,
        title=f"AARO Observing Nights — {month_label(period)}",
        x_limits=(month_start, month_end),
        x_breaks="3 days",
        x_labels="%d %b",
        figure_size=(12, 5),
    )
    save_plot(p, month_df, output_path, marker_size=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot night activity charts.")
    parser.add_argument("--csv", type=Path, default=NIGHT_REPORT_CSV)
    parser.add_argument(
        "--overview",
        type=Path,
        default=OVERVIEW_PLOT,
        help="Full longitudinal chart path",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        raise SystemExit(f"Report not found: {args.csv}. Run night_analysis.py first.")

    ensure_report_dirs()
    df = load_report(args.csv)

    render_overview(df, args.overview)
    print(f"Wrote {args.overview}")

    months = sorted(df["month"].unique())
    for period in months:
        output_path = monthly_plot_path(period)
        render_month(df, period, output_path)
        print(f"Wrote {output_path}")

    print(f"Generated overview + {len(months)} monthly charts under output/night-activity/plots/")


if __name__ == "__main__":
    main()
