#!/usr/bin/env python3
"""Per-sub FWHM scatter plot by observing session for any NINA LIGHT target."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from night_analysis import DEFAULT_LIGHT_ROOT, FILENAME_RE, night_label_for_frame, parse_frame
from report_paths import ensure_target_plot_dir, target_fwhm_plot_path, target_slug

DEFAULT_FWHM_THRESHOLD = 4.75
FILTER_ORDER = ["L", "R", "G", "B", "H", "O", "S", "Ha", "SII", "OIII"]
FILTER_COLORS = {
    "H": "#e377c2",
    "Ha": "#e377c2",
    "L": "#888888",
    "R": "#d62728",
    "G": "#2ca02c",
    "B": "#1f77b4",
    "O": "#ff7f0e",
    "S": "#9467bd",
    "SII": "#9467bd",
    "OIII": "#17becf",
}
FALLBACK_COLORS = plt.get_cmap("tab10").colors  # type: ignore[attr-defined]
DURATION_SIZE_ANCHORS = [(60, 18), (300, 32), (1200, 70), (1800, 110)]
FILTER_DODGE_DAYS = 0.012


def list_targets(light_root: Path) -> list[str]:
    return sorted(
        p.name
        for p in light_root.iterdir()
        if p.is_dir() and any(p.glob("*.fits"))
    )


def resolve_target_dir(light_root: Path, target: str) -> Path:
    direct = light_root / target
    if direct.is_dir():
        return direct

    slug = target_slug(target)
    matches = [
        p
        for p in light_root.iterdir()
        if p.is_dir() and target_slug(p.name) == slug
    ]
    if len(matches) == 1:
        return matches[0]

    available = list_targets(light_root)
    hint = ", ".join(available[:8])
    if len(available) > 8:
        hint += ", ..."
    raise SystemExit(
        f"Target not found: {target!r}. Available targets include: {hint}"
    )


def collect_target_frames(target_dir: Path, target_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fits_path in sorted(target_dir.glob("*.fits")):
        frame = parse_frame(fits_path, target_name)
        if frame is None:
            continue

        match = FILENAME_RE.match(fits_path.name)
        filt = match.group("filter") if match else "unknown"

        rows.append(
            {
                "session": pd.Timestamp(night_label_for_frame(frame)),
                "filter": filt,
                "fwhm": frame.fwhm,
                "duration_s": frame.duration_s,
                "duration_h": frame.duration_s / 3600.0,
                "filename": frame.filename,
                "is_bad": frame.filename.startswith("BAD_"),
            }
        )

    if not rows:
        raise SystemExit(f"No parsable FITS frames in {target_dir}")

    return pd.DataFrame(rows).sort_values(["session", "filter", "filename"]).reset_index(drop=True)


def filter_sort_key(name: str) -> tuple[int, str]:
    if name in FILTER_ORDER:
        return (FILTER_ORDER.index(name), name)
    return (len(FILTER_ORDER), name)


def filter_color(name: str) -> str:
    if name in FILTER_COLORS:
        return FILTER_COLORS[name]
    idx = abs(hash(name)) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[idx]  # type: ignore[return-value]


def duration_marker_size(duration_s: float) -> float:
    if duration_s <= DURATION_SIZE_ANCHORS[0][0]:
        return float(DURATION_SIZE_ANCHORS[0][1])
    if duration_s >= DURATION_SIZE_ANCHORS[-1][0]:
        return float(DURATION_SIZE_ANCHORS[-1][1])
    for (d0, s0), (d1, s1) in zip(DURATION_SIZE_ANCHORS, DURATION_SIZE_ANCHORS[1:]):
        if d0 <= duration_s <= d1:
            t = (duration_s - d0) / (d1 - d0)
            return s0 + t * (s1 - s0)
    return float(DURATION_SIZE_ANCHORS[-1][1])


def build_summary(df: pd.DataFrame, threshold: float) -> dict[str, object]:
    kept = df[df["fwhm"] < threshold]
    rejected = df[df["fwhm"] >= threshold]
    hours_by_filter = kept.groupby("filter")["duration_h"].sum().sort_index()
    return {
        "total_subs": len(df),
        "total_h": df["duration_h"].sum(),
        "kept_subs": len(kept),
        "kept_h": kept["duration_h"].sum(),
        "rejected_subs": len(rejected),
        "rejected_h": rejected["duration_h"].sum(),
        "hours_by_filter": hours_by_filter,
    }


def footer_text(summary: dict[str, object]) -> str:
    return (
        "Dot size = exposure duration · Filters grouped side-by-side · "
        f"{summary['total_subs']} subs · {summary['total_h']:.1f}h total · "
        f"Kept: {summary['kept_subs']} subs ({summary['kept_h']:.1f}h) · "
        f"Rejected: {summary['rejected_subs']} subs ({summary['rejected_h']:.1f}h)"
    )


def build_legend_handles(filters: list[str], threshold: float) -> list[Line2D]:
    filter_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=filter_color(filt),
            markersize=7,
            linestyle="None",
            label=filt,
        )
        for filt in filters
    ]
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#cccccc",
            markersize=np.sqrt(sz),
            linestyle="None",
            label=label,
        )
        for label, sz in [("< 60s", 18), ("300s", 32), ("1200s", 70), ("1800s", 110)]
    ]
    threshold_handle = Line2D(
        [0],
        [0],
        color="#cc0000",
        linestyle="--",
        linewidth=1.2,
        label=f"FWHM = {threshold:g}",
    )
    return filter_handles + size_handles + [threshold_handle]


def split_dataframe(
    df: pd.DataFrame,
    split_by: str,
) -> list[tuple[str, pd.DataFrame, pd.Timestamp, pd.Timestamp]]:
    if split_by == "year":
        grouped = df.groupby(df["session"].dt.year, sort=True)
        rows: list[tuple[str, pd.DataFrame, pd.Timestamp, pd.Timestamp]] = []
        for year, group in grouped:
            axis_start, axis_end = session_axis_bounds(sorted(group["session"].unique()))
            rows.append((str(year), group.reset_index(drop=True), axis_start, axis_end))
        return rows

    if split_by == "month":
        periods = df["session"].dt.to_period("M")
        grouped = df.groupby(periods, sort=True)
        rows = []
        for period, group in grouped:
            axis_start, axis_end = session_axis_bounds(sorted(group["session"].unique()))
            label = period.to_timestamp().strftime("%B %Y")
            rows.append((label, group.reset_index(drop=True), axis_start, axis_end))
        return rows

    raise ValueError(f"Unsupported split_by value: {split_by}")


def session_axis_bounds(sessions: list[pd.Timestamp]) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(sessions[0]).normalize(), pd.Timestamp(sessions[-1]).normalize()


def panel_y_limits(df: pd.DataFrame, threshold: float) -> tuple[float, float]:
    y_max = max(df["fwhm"].max(), threshold + 0.5)
    y_min = max(0, df["fwhm"].min() - 0.5)
    return y_min, y_max + 0.5


def session_date_num(session: pd.Timestamp) -> float:
    return float(mdates.date2num(pd.Timestamp(session).to_pydatetime()))


def filter_offsets(filters: list[str]) -> dict[str, float]:
    return {
        filt: (idx - (len(filters) - 1) / 2) * FILTER_DODGE_DAYS
        for idx, filt in enumerate(filters)
    }


def add_daily_vertical_grid(
    ax: plt.Axes,
    axis_start: pd.Timestamp,
    axis_end: pd.Timestamp,
    y_min: float,
    y_max: float,
) -> None:
    day = pd.Timestamp(axis_start).normalize()
    end = pd.Timestamp(axis_end).normalize()
    while day <= end:
        ax.vlines(
            session_date_num(day),
            y_min,
            y_max,
            colors="#ffffff",
            alpha=0.5,
            linewidth=0.65,
            zorder=2.5,
        )
        day += pd.Timedelta(days=1)


def configure_session_date_axis(
    ax: plt.Axes,
    axis_start: pd.Timestamp,
    axis_end: pd.Timestamp,
) -> None:
    start = pd.Timestamp(axis_start).normalize()
    end = pd.Timestamp(axis_end).normalize()
    ax.set_xlim(session_date_num(start) - 0.45, session_date_num(end) + 0.45)

    tick_days: list[pd.Timestamp] = []
    day = start
    while day <= end:
        tick_days.append(day)
        day += pd.Timedelta(days=1)

    ax.set_xticks([session_date_num(day) for day in tick_days])
    ax.set_xticklabels(
        [day.strftime("%Y-%m-%d") for day in tick_days],
        rotation=90,
        fontsize=8,
        color="#222222",
    )
    ax.grid(True, axis="y", which="major", color="#ffffff", alpha=0.55, linewidth=0.8)


def draw_fwhm_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    threshold: float,
    filters: list[str],
    *,
    panel_title: str | None = None,
    panel_title_in_axes: bool = False,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
    show_summary: bool = True,
    axis_start: pd.Timestamp | None = None,
    axis_end: pd.Timestamp | None = None,
) -> None:
    sessions = sorted(df["session"].unique())
    if not sessions:
        return

    if axis_start is None or axis_end is None:
        axis_start, axis_end = session_axis_bounds(sessions)

    dodge = filter_offsets(filters)
    y_min, y_max = panel_y_limits(df, threshold)
    x_right = session_date_num(axis_end) + 0.35

    ax.set_facecolor("#c8c8c8")
    ax.axhspan(threshold, y_max, color="#f4cccc", alpha=0.75, zorder=0)
    ax.axhspan(y_min, threshold, color="#e6e6e6", alpha=0.55, zorder=0)
    ax.axhline(threshold, color="#cc0000", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(
        x_right,
        threshold + 0.05,
        f"FWHM = {threshold:g}",
        color="#cc0000",
        fontsize=9,
        ha="right",
        va="bottom",
    )

    for _, row in df.iterrows():
        x = session_date_num(row["session"]) + dodge[row["filter"]]
        ax.scatter(
            x,
            row["fwhm"],
            s=duration_marker_size(row["duration_s"]),
            c=filter_color(row["filter"]),
            alpha=0.9 if not row["is_bad"] else 0.5,
            edgecolors="#333333" if not row["is_bad"] else "#999999",
            linewidths=0.35,
            zorder=3,
        )

    add_daily_vertical_grid(ax, axis_start, axis_end, y_min, y_max)

    configure_session_date_axis(ax, axis_start, axis_end)
    if show_xlabel:
        ax.set_xlabel("Session Night", color="#222222", labelpad=14)
    else:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=True)

    if show_ylabel:
        ax.set_ylabel("FWHM [arcsec]", color="#222222")
    else:
        ax.set_ylabel("")

    if panel_title:
        if panel_title_in_axes:
            ax.text(
                0.5,
                0.985,
                panel_title,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=11,
                fontweight="bold",
                color="#333333",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "#cccccc",
                },
                zorder=5,
            )
        else:
            ax.set_title(
                panel_title,
                color="#111111",
                fontsize=14,
                fontweight="bold",
                pad=14,
            )

    ax.tick_params(axis="x", colors="#222222", pad=4)
    ax.tick_params(axis="y", colors="#222222")
    for spine in ax.spines.values():
        spine.set_color("#888888")
    ax.set_ylim(y_min, y_max)

    if show_summary:
        summary = build_summary(df, threshold)
        hours_lines = [
            f"{filt}: {hours:.1f}h"
            for filt, hours in summary["hours_by_filter"].items()
        ]
        summary_text = (
            f"Kept: FWHM < {threshold:g}\n"
            + " · ".join(hours_lines)
            + f"\nTotal: {summary['kept_h']:.1f}h"
        )
        ax.text(
            0.01,
            0.98,
            summary_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="black",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.92},
        )


def place_footer(fig: plt.Figure, ax: plt.Axes, text: str) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_pos = ax.get_position()
    footer_x = (ax_pos.x0 + ax_pos.x1) / 2

    label_bottoms = [
        label.get_window_extent(renderer).y0
        for label in ax.get_xticklabels()
        if label.get_text()
    ]
    label_bottoms.append(ax.xaxis.label.get_window_extent(renderer).y0)
    content_bottom = min(label_bottoms) / fig.bbox.height
    footer_y = content_bottom - 0.028

    fig.text(
        footer_x,
        footer_y,
        text,
        ha="center",
        va="center",
        fontsize=8.5,
        color="#444444",
    )


def figure_width_for_range(axis_start: pd.Timestamp, axis_end: pd.Timestamp) -> float:
    span_days = (pd.Timestamp(axis_end).normalize() - pd.Timestamp(axis_start).normalize()).days + 1
    return max(12, span_days * 0.18 + 4)


def default_output_path(target_name: str, split_by: str) -> Path:
    base = target_fwhm_plot_path(target_name)
    if split_by == "none":
        return base
    return base.with_name(f"per-sub-fwhm-by-session-by-{split_by}.png")


def render_plot(
    df: pd.DataFrame,
    target_name: str,
    output_path: Path,
    threshold: float,
) -> None:
    filters = sorted(df["filter"].unique(), key=filter_sort_key)
    axis_start, axis_end = session_axis_bounds(sorted(df["session"].unique()))
    fig_w = figure_width_for_range(axis_start, axis_end)

    fig, ax = plt.subplots(figsize=(fig_w, 8.2), facecolor="white")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.30)

    draw_fwhm_panel(
        ax,
        df,
        threshold,
        filters,
        panel_title=f"{target_name} — Per-Sub FWHM by Session",
        show_xlabel=True,
        show_ylabel=True,
        show_summary=True,
    )

    legend = ax.legend(
        handles=build_legend_handles(filters, threshold),
        loc="upper right",
        framealpha=0.92,
        fontsize=8,
        title="Filters · Exposure · Threshold",
        title_fontsize=8,
    )
    legend.get_frame().set_facecolor("white")

    summary = build_summary(df, threshold)
    place_footer(fig, ax, footer_text(summary))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor="white")
    plt.close(fig)


def render_split_plot(
    df: pd.DataFrame,
    target_name: str,
    output_path: Path,
    threshold: float,
    split_by: str,
) -> None:
    groups = split_dataframe(df, split_by)
    filters = sorted(df["filter"].unique(), key=filter_sort_key)
    fig_w = max(figure_width_for_range(start, end) for _, _, start, end in groups)
    max_days = max((end - start).days + 1 for _, _, start, end in groups)
    row_h = max(3.2, min(5.0, max_days * 0.04 + 2.8))
    fig_h = row_h * len(groups) + 2.2

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(fig_w, fig_h),
        facecolor="white",
        squeeze=False,
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.93, bottom=0.12, hspace=0.50)
    fig.suptitle(
        f"{target_name} — Per-Sub FWHM by Session (by {split_by})",
        color="#111111",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    flat_axes = axes.flatten()
    for ax, (label, group, axis_start, axis_end) in zip(flat_axes, groups):
        draw_fwhm_panel(
            ax,
            group,
            threshold,
            filters,
            panel_title=label,
            panel_title_in_axes=True,
            show_xlabel=False,
            show_ylabel=True,
            show_summary=True,
            axis_start=axis_start,
            axis_end=axis_end,
        )

    bottom_ax = flat_axes[-1]
    bottom_ax.set_xlabel("Session Night", color="#222222", labelpad=14)

    legend = bottom_ax.legend(
        handles=build_legend_handles(filters, threshold),
        loc="upper right",
        framealpha=0.92,
        fontsize=8,
        title="Filters · Exposure · Threshold",
        title_fontsize=8,
    )
    legend.get_frame().set_facecolor("white")

    summary = build_summary(df, threshold)
    place_footer(fig, bottom_ax, footer_text(summary))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-sub FWHM by session for a NINA LIGHT target."
    )
    parser.add_argument(
        "--target",
        help="Target folder name under LIGHT (e.g. 'M81 and M82')",
    )
    parser.add_argument(
        "--light-root",
        type=Path,
        default=DEFAULT_LIGHT_ROOT,
        help=f"NINA LIGHT root (default: {DEFAULT_LIGHT_ROOT})",
    )
    parser.add_argument(
        "--light-dir",
        type=Path,
        help="Direct path to a target LIGHT folder (overrides --target)",
    )
    parser.add_argument(
        "--fwhm-threshold",
        type=float,
        default=DEFAULT_FWHM_THRESHOLD,
        help=f"FWHM keep threshold in arcsec (default: {DEFAULT_FWHM_THRESHOLD})",
    )
    parser.add_argument(
        "--split-by",
        choices=["none", "year", "month"],
        default="none",
        help="Break the session axis into stacked subplots by year or month",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path (default depends on --split-by)",
    )
    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List available targets under --light-root and exit",
    )
    args = parser.parse_args()

    if not args.light_root.is_dir() and not args.light_dir:
        raise SystemExit(f"LIGHT root not found: {args.light_root}")

    if args.list_targets:
        for name in list_targets(args.light_root):
            print(name)
        return

    if args.light_dir:
        target_dir = args.light_dir
        target_name = args.target or target_dir.name
    elif args.target:
        target_dir = resolve_target_dir(args.light_root, args.target)
        target_name = target_dir.name
    else:
        raise SystemExit("Provide --target, --light-dir, or --list-targets")

    output = args.output or default_output_path(target_name, args.split_by)
    ensure_target_plot_dir(target_name)

    df = collect_target_frames(target_dir, target_name)
    if args.split_by == "none":
        render_plot(df, target_name, output, args.fwhm_threshold)
    else:
        render_split_plot(df, target_name, output, args.fwhm_threshold, args.split_by)

    print(f"Parsed {len(df)} subs across {df['session'].nunique()} sessions")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
