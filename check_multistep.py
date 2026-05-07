"""
Scan ERA5 files and generate valid_windows.json for gap-aware multi-step training.

This script:
1. Reads all NetCDF/Zarr files in the input directory
2. For each file, finds all start indices where the entire window of length
   (history_len + forecast_len + 1) has uniform expected_interval_h spacing
3. Outputs valid_windows.json containing only window start positions that
   guarantee a fully gap-free rollout

Usage:
    python check_multistep.py \
        --input_files "/path/to/era5/y*.nc" \
        --history_len 1 \
        --forecast_len 6 \
        --expected_interval_h 6

    # Output filename auto-includes the params:
    #   valid_windows_h1_f6.json
    #   valid_windows_h1_f6_timestamps.json
    #   valid_windows_h1_f6_stats.json

    # Or specify multiple files explicitly:
    python check_multistep.py \
        --input_files /path/to/y2020.nc /path/to/y2021.nc \
        --history_len 1 --forecast_len 6 \
        --expected_interval_h 6
"""

import argparse
import json
import glob
import os
import random
import numpy as np
import xarray as xr
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def open_dataset(filename):
    """Open NetCDF or Zarr file."""
    if filename.endswith(".nc") or filename.endswith(".nc4"):
        return xr.open_dataset(filename)
    else:
        return xr.open_zarr(filename, chunks=None)


def scan_valid_windows(filenames, history_len, forecast_len, expected_interval_h=6):
    """
    Scan all files and find every start index where a window of
    (history_len + forecast_len + 1) consecutive timesteps has uniform spacing.

    Window logic mirrors the parent dataset class:
        ind_start_in_file ... ind_start_in_file + history_len + forecast_len
        slice(ind_start, ind_end + 1) -> length = history_len + forecast_len + 1
        number of adjacent diffs to check = history_len + forecast_len

    Args:
        filenames: list of sorted file paths
        history_len: number of input history steps
        forecast_len: number of forecast (rollout) steps
        expected_interval_h: expected interval in hours (default: 6)

    Returns:
        valid_windows: list of {"file_id": int, "start_time_idx": int}
        all_timestamps: dict of {file_id: {"filename", "n_times", "timestamps"}}
        stats: dict with summary statistics including gap-type histogram
    """
    expected_interval = np.timedelta64(expected_interval_h, "h")
    window_n_diffs = history_len + forecast_len  # diffs to check per window
    window_size = window_n_diffs + 1              # actual timesteps per window

    valid_windows = []
    all_timestamps = {}

    total_possible = 0
    total_valid = 0
    total_dropped = 0
    global_gap_summary = {}

    for file_id, fn in enumerate(filenames):
        ds = open_dataset(fn)
        times = ds["time"].values
        n_times = len(times)

        all_timestamps[file_id] = {
            "filename": os.path.basename(fn),
            "n_times": n_times,
            "timestamps": [
                str(np.datetime_as_string(t, unit="s")) for t in times
            ],
        }

        # Skip files that are shorter than one window
        if n_times < window_size:
            logger.warning(
                f"File {file_id}: {os.path.basename(fn)} has only "
                f"{n_times} timesteps (< window size {window_size}). Skipping."
            )
            ds.close()
            continue

        # Number of possible window start positions in this file
        n_possible = n_times - window_n_diffs  # equiv. n_times - window_size + 1

        # ---- cumsum-based sliding check ---- #
        # diffs[i] = times[i+1] - times[i],  shape (n_times - 1,)
        diffs = np.diff(times)
        is_good = (diffs == expected_interval)

        # n_bad_in_window[i] = count of bad diffs in diffs[i : i + window_n_diffs]
        # which corresponds to checking times[i : i + window_size]
        cumsum = np.concatenate([[0], np.cumsum(~is_good).astype(np.int64)])
        n_bad_in_window = cumsum[window_n_diffs:] - cumsum[:-window_n_diffs]
        valid_starts = np.where(n_bad_in_window == 0)[0]

        for s in valid_starts:
            valid_windows.append({
                "file_id": file_id,
                "start_time_idx": int(s),
            })

        # ---- per-file gap-type histogram ---- #
        file_gap_summary = {}
        bad_diffs = diffs[~is_good]
        for d in bad_diffs:
            d_h = d / np.timedelta64(1, "h")
            key = f"{d_h:.1f}h"
            file_gap_summary[key] = file_gap_summary.get(key, 0) + 1
            global_gap_summary[key] = global_gap_summary.get(key, 0) + 1

        file_valid = len(valid_starts)
        file_dropped = n_possible - file_valid
        total_possible += n_possible
        total_valid += file_valid
        total_dropped += file_dropped

        ds.close()

        logger.info(
            f"File {file_id}: {os.path.basename(fn)} | "
            f"{n_times} timesteps | "
            f"{file_valid}/{n_possible} valid windows | "
            f"dropped {file_dropped} | "
            f"gap types: {file_gap_summary if file_gap_summary else 'none'}"
        )

    drop_rate_pct = total_dropped / max(total_possible, 1) * 100

    stats = {
        "total_files": len(filenames),
        "history_len": history_len,
        "forecast_len": forecast_len,
        "window_size": window_size,
        "window_n_diffs": window_n_diffs,
        "expected_interval_h": expected_interval_h,
        "total_possible_windows": total_possible,
        "total_valid_windows": total_valid,
        "total_dropped": total_dropped,
        "drop_rate": f"{drop_rate_pct:.2f}%",
        "global_gap_summary": global_gap_summary,
        "filenames": [os.path.basename(f) for f in filenames],
    }

    return valid_windows, all_timestamps, stats


def sanity_check(valid_windows, all_timestamps, history_len, forecast_len,
                 expected_interval_h, n_samples=5):
    """
    Randomly sample a few valid windows and print the full timestamp sequence
    so the user can eyeball-verify they really are gap-free.
    """
    if len(valid_windows) == 0:
        logger.warning("No valid windows found, skipping sanity check.")
        return

    expected_interval = np.timedelta64(expected_interval_h, "h")
    window_size = history_len + forecast_len + 1

    logger.info("===== Sanity check (random samples) =====")
    samples = random.sample(valid_windows, min(n_samples, len(valid_windows)))
    for w in samples:
        fid = w["file_id"]
        s = w["start_time_idx"]
        ts = all_timestamps[fid]["timestamps"][s : s + window_size]
        # Re-verify in case of any logic error
        ts_np = np.array(ts, dtype="datetime64[s]")
        diffs = np.diff(ts_np)
        all_ok = bool(np.all(diffs == expected_interval))
        marker = "OK" if all_ok else "FAIL"
        logger.info(
            f"  [{marker}] file_id={fid} start={s} "
            f"file={all_timestamps[fid]['filename']}"
        )
        for t in ts:
            logger.info(f"      {t}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate valid_windows.json for gap-aware multi-step dataset"
    )
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="File paths or glob pattern. E.g.: '/data/y*.nc' or y2020.nc y2021.nc",
    )
    parser.add_argument(
        "--history_len",
        type=int,
        required=True,
        help="Number of input history timesteps (must match training config)",
    )
    parser.add_argument(
        "--forecast_len",
        type=int,
        required=True,
        help="Number of forecast (rollout) timesteps (must match training config)",
    )
    parser.add_argument(
        "--expected_interval_h",
        type=int,
        default=6,
        help="Expected time interval in hours (default: 6)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory (default: current dir)",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="valid_windows",
        help="Output filename prefix (default: valid_windows)",
    )
    parser.add_argument(
        "--warn_drop_rate_pct",
        type=float,
        default=10.0,
        help="Emit a warning if drop rate exceeds this percentage (default: 10.0)",
    )
    parser.add_argument(
        "--n_sanity_samples",
        type=int,
        default=5,
        help="How many random valid windows to print for sanity check (default: 5)",
    )
    args = parser.parse_args()

    # Resolve file paths (handle glob patterns)
    filenames = []
    for pattern in args.input_files:
        expanded = sorted(glob.glob(pattern))
        if expanded:
            filenames.extend(expanded)
        else:
            # Not a glob, treat as literal path
            filenames.append(pattern)

    filenames = sorted(filenames)
    logger.info(f"Found {len(filenames)} files")
    for fn in filenames:
        logger.info(f"  {fn}")

    # Build output paths with params baked in
    base = (
        f"{args.output_prefix}"
        f"_h{args.history_len}"
        f"_f{args.forecast_len}"
        f"_dt{args.expected_interval_h}h"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{base}.json")
    timestamps_path = os.path.join(args.output_dir, f"{base}_timestamps.json")
    stats_path = os.path.join(args.output_dir, f"{base}_stats.json")

    # Scan
    valid_windows, all_timestamps, stats = scan_valid_windows(
        filenames,
        history_len=args.history_len,
        forecast_len=args.forecast_len,
        expected_interval_h=args.expected_interval_h,
    )

    # Save valid_windows.json
    with open(output_path, "w") as f:
        json.dump(valid_windows, f)
    logger.info(f"Saved {len(valid_windows)} valid windows to {output_path}")

    # Save all_timestamps.json
    with open(timestamps_path, "w") as f:
        json.dump(all_timestamps, f, indent=2)
    logger.info(f"Saved all timestamps to {timestamps_path}")

    # Save stats
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats to {stats_path}")

    # Sanity check
    sanity_check(
        valid_windows,
        all_timestamps,
        history_len=args.history_len,
        forecast_len=args.forecast_len,
        expected_interval_h=args.expected_interval_h,
        n_samples=args.n_sanity_samples,
    )

    # Summary
    logger.info("===== Summary =====")
    logger.info(f"Total files:         {stats['total_files']}")
    logger.info(f"history_len:         {stats['history_len']}")
    logger.info(f"forecast_len:        {stats['forecast_len']}")
    logger.info(f"Window size:         {stats['window_size']} timesteps")
    logger.info(f"Expected interval:   {stats['expected_interval_h']}h")
    logger.info(f"Possible windows:    {stats['total_possible_windows']}")
    logger.info(f"Valid windows:       {stats['total_valid_windows']}")
    logger.info(f"Dropped windows:     {stats['total_dropped']}")
    logger.info(f"Drop rate:           {stats['drop_rate']}")
    logger.info(f"Gap-type histogram:  {stats['global_gap_summary']}")

    drop_rate_pct = float(stats["drop_rate"].rstrip("%"))
    if drop_rate_pct > args.warn_drop_rate_pct:
        logger.warning(
            f"Drop rate {drop_rate_pct:.2f}% exceeds threshold "
            f"{args.warn_drop_rate_pct:.2f}%. "
            f"Your data has more gaps than typical. "
            f"Inspect the gap-type histogram and timestamps file before training."
        )


if __name__ == "__main__":
    main()