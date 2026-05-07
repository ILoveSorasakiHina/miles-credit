"""
Scan ERA5 files and generate valid_pairs.json for gap-aware training.

This script:
1. Reads all NetCDF/Zarr files in the input directory
2. Checks every adjacent time step pair for the expected interval
3. Outputs valid_pairs.json containing only pairs with correct spacing

Usage:
    python check.py \
        --input_files "/path/to/era5/y*.nc" \
        --output_path ./valid_pairs.json \
        --expected_interval_h 1

    # Or specify multiple files explicitly:
    python check.py \
        --input_files /path/to/y2020.nc /path/to/y2021.nc /path/to/y2022.nc \
        --output_path ./valid_pairs.json \
        --expected_interval_h 1
"""

import argparse
import json
import glob
import os
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


def scan_valid_pairs(filenames, expected_interval_h=1):
    """
    Scan all files and find valid (input, target) pairs
    where the time gap equals exactly expected_interval_h.

    Args:
        filenames: list of sorted file paths
        expected_interval_h: expected interval in hours (default: 1)

    Returns:
        valid_pairs: list of {"file_id": int, "time_idx": int}
        all_timestamps: dict of {file_id: {"filename", "n_times", "timestamps"}}
        stats: dict with summary statistics
    """
    expected_interval = np.timedelta64(expected_interval_h, "h")
    valid_pairs = []
    all_timestamps = {}

    total_timesteps = 0
    total_gaps = 0
    total_valid = 0

    for file_id, fn in enumerate(filenames):
        ds = open_dataset(fn)
        times = ds["time"].values
        n_times = len(times)
        total_timesteps += n_times

        # Save all timestamps as strings for inspection
        all_timestamps[file_id] = {
            "filename": os.path.basename(fn),
            "n_times": n_times,
            "timestamps": [
                str(np.datetime_as_string(t, unit="s")) for t in times
            ],
        }

        # Check every adjacent pair
        file_valid = 0
        file_gaps = 0

        for i in range(n_times - 1):
            dt = times[i + 1] - times[i]

            if dt == expected_interval:
                valid_pairs.append({
                    "file_id": file_id,
                    "time_idx": i,
                })
                file_valid += 1
            else:
                file_gaps += 1
                # Log the first few gaps per file for debugging
                if file_gaps <= 5:
                    logger.debug(
                        f"  Gap at file {file_id}, index {i}: "
                        f"{times[i]} -> {times[i+1]} "
                        f"(interval={dt}, expected={expected_interval})"
                    )

        total_valid += file_valid
        total_gaps += file_gaps

        ds.close()

        logger.info(
            f"File {file_id}: {os.path.basename(fn)} | "
            f"{n_times} timesteps | "
            f"{file_valid} valid pairs | "
            f"{file_gaps} gaps found"
        )

    stats = {
        "total_files": len(filenames),
        "total_timesteps": total_timesteps,
        "total_valid_pairs": total_valid,
        "total_gaps": total_gaps,
        "drop_rate": f"{total_gaps / max(total_timesteps - len(filenames), 1) * 100:.2f}%",
        "expected_interval_h": expected_interval_h,
        "filenames": [os.path.basename(f) for f in filenames],
    }

    return valid_pairs, all_timestamps, stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate valid_pairs.json for gap-aware dataset"
    )
    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="File paths or glob pattern. E.g.: '/data/y*.nc' or y2020.nc y2021.nc",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./valid_pairs.json",
        help="Output path for valid_pairs.json",
    )
    parser.add_argument(
        "--expected_interval_h",
        type=int,
        default=1,
        help="Expected time interval in hours (default: 1)",
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

    # Scan
    valid_pairs, all_timestamps, stats = scan_valid_pairs(
        filenames,
        expected_interval_h=args.expected_interval_h,
    )

    # Save valid_pairs.json
    with open(args.output_path, "w") as f:
        json.dump(valid_pairs, f)
    logger.info(f"Saved {len(valid_pairs)} valid pairs to {args.output_path}")

    # Save all_timestamps.json
    timestamps_path = args.output_path.replace(".json", "_timestamps.json")
    with open(timestamps_path, "w") as f:
        json.dump(all_timestamps, f, indent=2)
    logger.info(f"Saved all timestamps to {timestamps_path}")

    # Save stats
    stats_path = args.output_path.replace(".json", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved stats to {stats_path}")

    # Print summary
    logger.info("\n===== Summary =====")
    logger.info(f"Total files:       {stats['total_files']}")
    logger.info(f"Total timesteps:   {stats['total_timesteps']}")
    logger.info(f"Valid pairs:       {stats['total_valid_pairs']}")
    logger.info(f"Gaps found:        {stats['total_gaps']}")
    logger.info(f"Gap rate:          {stats['drop_rate']}")


if __name__ == "__main__":
    main()