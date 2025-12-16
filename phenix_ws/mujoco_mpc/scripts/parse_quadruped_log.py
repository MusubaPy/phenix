#!/usr/bin/env python3
"""Parse and plot quadruped CSV log produced by mjpc."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def read_csv(path: Path) -> Dict[str, List[float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        data: Dict[str, List[float]] = {name: [] for name in reader.fieldnames}
        for row in reader:
            for key, val in row.items():
                try:
                    data[key].append(float(val))
                except (TypeError, ValueError):
                    data[key].append(float("nan"))
    return data


def plot_series(ax, time, keys, data, title, ylabel=""):
    for key in keys:
        if key in data:
            ax.plot(time, data[key], label=key, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("time [s]")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(keys) > 1:
        ax.legend(loc="best", fontsize="small")


def collect_keys(data: Dict[str, List[float]], prefix: str):
    return [k for k in data.keys() if k.startswith(prefix)]


def main():
    parser = argparse.ArgumentParser(description="Plot quadruped CSV log.")
    parser.add_argument(
        "csv_path",
        nargs="?",
        default="logs/quadruped_log.csv",
        help="Path to CSV log (default: logs/quadruped_log.csv)",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Start time [s] to plot from (rows with time < tmin are dropped)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save figures (if not set, plots are shown interactively)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.is_file():
        print(f"[ERROR] CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    try:
        data = read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - basic error path
        print(f"[ERROR] Failed to read CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    if "time" not in data:
        print("[ERROR] CSV missing 'time' column", file=sys.stderr)
        sys.exit(1)

    time = data["time"]

    if args.tmin is not None:
        mask = [t >= args.tmin for t in time]
        if not any(mask):
            print(f"[ERROR] No samples at t >= {args.tmin}", file=sys.stderr)
            sys.exit(1)
        def apply_mask(seq):
            return [v for v, keep in zip(seq, mask) if keep]
        time = apply_mask(time)
        for key in list(data.keys()):
            data[key] = apply_mask(data[key]) if len(data[key]) == len(mask) else data[key]

    torque_error_keys = collect_keys(data, "torque_error_")
    torque_cmd_keys = collect_keys(data, "torque_cmd_")
    torque_applied_keys = collect_keys(data, "torque_applied_")
    vel_keys = collect_keys(data, "joint_vel_")
    grf_keys = collect_keys(data, "grf_")

    backend = plt.get_backend().lower()
    non_interactive = "agg" in backend

    figs = []

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_series(ax, time, torque_error_keys, data, "Torque error (cmd - applied)", "torque [Nm]")
    figs.append((fig, "torque_error"))

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_series(ax, time, torque_applied_keys, data, "Applied torques", "torque [Nm]")
    figs.append((fig, "torque_applied"))

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_series(ax, time, vel_keys, data, "Joint angular velocities", "rad/s")
    figs.append((fig, "joint_vel"))

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_series(ax, time, [k for k in data.keys() if k in ("com_x", "com_y", "com_z")], data, "Center of mass", "meters")
    figs.append((fig, "com"))

    if grf_keys:
        # Group GRF by foot
        feet = {"FL": [], "FR": [], "HL": [], "HR": []}
        for key in grf_keys:
            for foot in feet:
                if f"grf_{foot}_" in key:
                    feet[foot].append(key)
                    break
        for foot, keys in feet.items():
            if not keys:
                continue
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_series(ax, time, sorted(keys), data, f"GRF {foot}", "force [N]")
            figs.append((fig, f"grf_{foot}"))

    outdir: Path | None = None
    if args.outdir:
        outdir = Path(args.outdir)
    elif non_interactive:
        outdir = csv_path.parent / "plots"
        print(
            f"[INFO] Non-interactive backend detected ({backend}); "
            f"saving figures to {outdir}"
        )

    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)
        for fig, name in figs:
            fig.savefig(outdir / f"{name}.png", dpi=200, bbox_inches="tight")
        print(f"Saved {len(figs)} figure(s) to {outdir}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
