"""
Gap-aware dataset for irregular time steps.

Inherits from ERA5_and_Forcing_SingleStep, only overrides indexing logic.
All data reading, merging, transform logic remains untouched.

Usage:
    1. Run the preprocessing script to generate valid_pairs.json
    2. Point this dataset to that file via `valid_pairs_path` argument
    3. Everything else works the same as ERA5_and_Forcing_SingleStep
"""

import json
import logging
import numpy as np
from credit.datasets.era5_singlestep import ERA5_and_Forcing_SingleStep

logger = logging.getLogger(__name__)


class GapAwareSingleStep(ERA5_and_Forcing_SingleStep):
    """
    A single-step dataset that skips time pairs with irregular gaps.

    Instead of assuming all adjacent time indices are valid 1h pairs,
    this dataset reads a pre-computed index table that maps
    dataset index -> (file_id, time_index_in_file).

    Only 3 things change from the parent class:
        1. __init__: loads the valid pairs index table
        2. __len__: returns the number of valid pairs
        3. __getitem__: maps index -> (file_id, time_idx), then
           calls the parent's data loading logic
    """

    def __init__(self, *args, valid_pairs_path=None, **kwargs):
        """
        Args:
            valid_pairs_path (str): Path to JSON file containing valid pairs.
                Format: [{"file_id": 0, "time_idx": 5}, ...]
                Each entry means: input=time[time_idx], target=time[time_idx+1]
                and the gap between them is exactly the expected interval (e.g., 1h).

            All other args/kwargs are passed to ERA5_and_Forcing_SingleStep.
        """
        # Call parent __init__ (opens files, builds ERA5_indices, etc.)
        super().__init__(*args, **kwargs)

        # Load the valid pairs index table
        if valid_pairs_path is None:
            raise ValueError(
                "valid_pairs_path is required. "
                "Run the preprocessing script first to generate this file."
            )

        with open(valid_pairs_path, "r") as f:
            self.valid_pairs = json.load(f)

        logger.info(
            f"Loaded {len(self.valid_pairs)} valid pairs "
            f"(original dataset had {super().__len__()} indices)"
        )

    def __len__(self):
        """Number of valid (input, target) pairs."""
        return len(self.valid_pairs)

    def __getitem__(self, index):
        """
        Map index to a valid (file_id, time_idx) pair,
        then use the parent class's data loading logic.

        The parent's __getitem__ expects a global index that maps to
        a position across all files. We reconstruct that global index
        from (file_id, time_idx) using ERA5_indices.
        """

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        logger.setLevel(logging.INFO)
        
        pair = self.valid_pairs[index]
        file_id = pair["file_id"]
        time_idx = pair["time_idx"]

        # Convert (file_id, time_idx) back to the global index
        # that the parent class expects.
        # ERA5_indices[file_id] = [n_times, global_start, global_end]
        global_start = self.ERA5_indices[str(file_id)][1]
        global_index = global_start + time_idx

        # Log the mapping before calling parent
        # Get actual timestamps from the file for verification
        times = self.all_files[file_id]["time"].values
        input_time = str(np.datetime_as_string(times[time_idx], unit="s"))
        target_time = str(np.datetime_as_string(times[time_idx + 1], unit="s"))
        dt_hours = (times[time_idx + 1] - times[time_idx]) / np.timedelta64(1, "h")

        if dt_hours != 6:
            raise ValueError(
                f"[GapAware] INVALID PAIR! index={index}, file_id={file_id}, "
                f"time_idx={time_idx}, input={input_time}, target={target_time}, "
                f"dt={dt_hours:.1f}h (expected 6.0h)"
            )


        # Call parent's __getitem__ with the correct global index
        sample = super().__getitem__(global_index)

        expected_input_ts = int(times[time_idx].astype("datetime64[s]").astype(int))
        expected_target_ts = int(times[time_idx + 1].astype("datetime64[s]").astype(int))
        actual_input_ts = sample['datetime'][0]
        actual_target_ts = sample['datetime'][1]

        if expected_input_ts != actual_input_ts or expected_target_ts != actual_target_ts:
            raise ValueError(
                f"[GapAware] TIME MISMATCH! index={index}, "
                f"expected=({input_time}, {target_time}), "
                f"actual=({actual_input_ts}, {actual_target_ts})"
            )



        return sample