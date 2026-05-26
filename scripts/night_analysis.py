#!/usr/bin/env python3
"""Reconstruct observing nights from NINA LIGHT frame filenames.

Walks all target subdirectories, parses timestamps and exposure durations from
filenames, groups frames into nights, detects roof-closure gaps (>= 30 min),
and writes a per-night summary CSV plus a terminal table.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from astroplan import Observer
from astropy.coordinates import EarthLocation
from astropy.time import Time
from tabulate import tabulate

import astropy.units as u

# AARO observatory, Rodeo NM (from AcquisitionDetails.csv)
OBS_LAT = 31.9072
OBS_LON = -109.0208
OBS_ELEV_M = 1257.0
OBS_TIMEZONE = ZoneInfo("America/Denver")

DEFAULT_LIGHT_ROOT = Path("/Volumes/FileStore/ASTRO/AARO/NINA/LIGHT")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "night_report.csv"

NIGHT_SPLIT_GAP = timedelta(hours=12)
ROOF_GAP_THRESHOLD = timedelta(minutes=30)

FILENAME_RE = re.compile(
    r"^(?:BAD_)?"
    r"DATE_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"TIME_(?P<time>\d{2}-\d{2}-\d{2})_"
    r"FILTER_(?P<filter>[^_]+)_"
    r"ANGLE_(?P<angle>[\d.]+)_"
    r"FWHM_(?P<fwhm>[\d.]+)_"
    r"TEMP_(?P<temp>[-\d.]+)_"
    r"(?P<duration>[\d.]+)s_"
    r"(?P<seq>\d+)\.fits$"
)


@dataclass(frozen=True)
class Frame:
    start: datetime
    duration_s: float
    fwhm: float
    target_dir: str
    filename: str

    @property
    def end(self) -> datetime:
        return self.start + timedelta(seconds=self.duration_s)


@dataclass
class NightMetrics:
    night_date: date
    sunset: datetime
    sunrise: datetime
    dark_window_hrs: float
    avg_fwhm: float
    imaging_hrs: float
    roof_closure_count: int
    total_closure_hrs: float
    pct_active: float


def parse_frame(path: Path, target_dir: str) -> Frame | None:
    match = FILENAME_RE.match(path.name)
    if not match:
        return None

    start = datetime.strptime(
        f"{match.group('date')} {match.group('time').replace('-', ':')}",
        "%Y-%m-%d %H:%M:%S",
    ).replace(tzinfo=OBS_TIMEZONE)

    return Frame(
        start=start,
        duration_s=float(match.group("duration")),
        fwhm=float(match.group("fwhm")),
        target_dir=target_dir,
        filename=path.name,
    )


def collect_frames(light_root: Path) -> list[Frame]:
    frames: list[Frame] = []
    for target_path in sorted(light_root.iterdir()):
        if not target_path.is_dir():
            continue
        target_name = target_path.name
        for fits_path in target_path.glob("*.fits"):
            frame = parse_frame(fits_path, target_name)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda f: f.start)
    return frames


def night_label_for_frame(frame: Frame) -> date:
    """Evening calendar date for the observing night."""
    local_start = frame.start.astimezone(OBS_TIMEZONE)
    if local_start.hour < 12:
        return (local_start.date() - timedelta(days=1))
    return local_start.date()


def split_into_nights(frames: list[Frame]) -> list[list[Frame]]:
    if not frames:
        return []

    nights: list[list[Frame]] = [[frames[0]]]
    for frame in frames[1:]:
        prev = nights[-1][-1]
        gap = frame.start - prev.end
        if gap > NIGHT_SPLIT_GAP:
            nights.append([frame])
        else:
            nights[-1].append(frame)
    return nights


def label_night(frames: list[Frame]) -> date:
    labels = [night_label_for_frame(f) for f in frames]
    return max(set(labels), key=labels.count)


def detect_roof_gaps(frames: list[Frame]) -> tuple[int, float]:
    closure_count = 0
    total_closure_s = 0.0

    for prev, nxt in zip(frames, frames[1:]):
        gap = nxt.start - prev.end
        if gap >= ROOF_GAP_THRESHOLD:
            closure_count += 1
            total_closure_s += gap.total_seconds()

    return closure_count, total_closure_s / 3600.0


def build_observer() -> Observer:
    location = EarthLocation(
        lat=OBS_LAT * u.deg,
        lon=OBS_LON * u.deg,
        height=OBS_ELEV_M * u.m,
    )
    return Observer(location=location, timezone=str(OBS_TIMEZONE))


def twilight_times(observer: Observer, night_date: date) -> tuple[datetime, datetime]:
    """Return evening and morning astronomical twilight (-18 deg) for a night."""
    noon = datetime(
        night_date.year,
        night_date.month,
        night_date.day,
        12,
        0,
        0,
        tzinfo=OBS_TIMEZONE,
    )
    t = Time(noon)

    dusk = observer.twilight_evening_astronomical(t, which="next")
    dawn = observer.twilight_morning_astronomical(t, which="next")

    sunset = dusk.to_datetime(timezone=OBS_TIMEZONE)
    sunrise = dawn.to_datetime(timezone=OBS_TIMEZONE)
    return sunset, sunrise


def analyze_night(observer: Observer, frames: list[Frame]) -> NightMetrics:
    night_date = label_night(frames)
    sunset, sunrise = twilight_times(observer, night_date)

    dark_window_hrs = max((sunrise - sunset).total_seconds(), 0.0) / 3600.0
    imaging_hrs = sum(f.duration_s for f in frames) / 3600.0
    avg_fwhm = float(np.mean([f.fwhm for f in frames]))
    roof_closure_count, total_closure_hrs = detect_roof_gaps(frames)

    pct_active = (imaging_hrs / dark_window_hrs * 100.0) if dark_window_hrs > 0 else 0.0

    return NightMetrics(
        night_date=night_date,
        sunset=sunset,
        sunrise=sunrise,
        dark_window_hrs=dark_window_hrs,
        avg_fwhm=avg_fwhm,
        imaging_hrs=imaging_hrs,
        roof_closure_count=roof_closure_count,
        total_closure_hrs=total_closure_hrs,
        pct_active=pct_active,
    )


def fmt_time(dt: datetime) -> str:
    return dt.astimezone(OBS_TIMEZONE).strftime("%Y-%m-%d %H:%M")


def metrics_to_row(m: NightMetrics) -> dict[str, str | float | int]:
    return {
        "Date": m.night_date.isoformat(),
        "Sunset (astro twilight)": fmt_time(m.sunset),
        "Sunrise (astro twilight)": fmt_time(m.sunrise),
        "Dark window (hrs)": round(m.dark_window_hrs, 2),
        "Avg FWHM": round(m.avg_fwhm, 2),
        "Imaging time (hrs)": round(m.imaging_hrs, 2),
        "Roof closure count": m.roof_closure_count,
        "Total closure time (hrs)": round(m.total_closure_hrs, 2),
        "% active": round(m.pct_active, 1),
    }


def write_csv(rows: list[dict[str, str | float | int]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze NINA LIGHT observing nights.")
    parser.add_argument(
        "--light-root",
        type=Path,
        default=DEFAULT_LIGHT_ROOT,
        help=f"Path to NINA LIGHT directory (default: {DEFAULT_LIGHT_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    if not args.light_root.is_dir():
        raise SystemExit(f"LIGHT root not found: {args.light_root}")

    frames = collect_frames(args.light_root)
    if not frames:
        raise SystemExit(f"No parsable FITS frames found under {args.light_root}")

    observer = build_observer()
    nights = split_into_nights(frames)
    metrics = [analyze_night(observer, night_frames) for night_frames in nights]
    metrics.sort(key=lambda m: m.night_date)

    rows = [metrics_to_row(m) for m in metrics]
    write_csv(rows, args.output)

    print(f"Parsed {len(frames)} frames across {len(metrics)} nights")
    print(f"Wrote {args.output}\n")
    print(tabulate(rows, headers="keys", tablefmt="simple"))


if __name__ == "__main__":
    main()
