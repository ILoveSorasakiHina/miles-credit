"""
Gap-aware multi-step dataset for irregular / missing timesteps.

Inherits from ERA5_MultiStep_Batcher; only overrides indexing so that the
underlying DistributedSampler can only ever pick window start positions that
are guaranteed to be gap-free for the full (history_len + forecast_len + 1)
rollout horizon.

Usage:
    1. Run `check_multistep.py` to generate
       `valid_windows_h{H}_f{F}_dt{N}h.json` for the desired (history_len,
       forecast_len, expected_interval_h).
    2. Point this dataset to that file via `valid_windows_path` argument.
    3. Everything else (worker, transforms, batching, rollout) behaves
       exactly the same as ERA5_MultiStep_Batcher.

JSON format (produced by check_multistep.py):
    [{"file_id": 0, "start_time_idx": 13}, ...]
"""

import json
import logging
import numpy as np
from torch.utils.data import DistributedSampler

from credit.datasets.era5_multistep_batcher import ERA5_MultiStep_Batcher

logger = logging.getLogger(__name__)


class GapAwareMultiStepBatcher(ERA5_MultiStep_Batcher):
    """
    A multi-step batcher that only samples from windows known to be gap-free.

    The parent class lets DistributedSampler pick any global index in
    [0, total_len), assuming every position can start a full
    (history_len + forecast_len + 1) window with the expected interval.

    When the underlying zarr/nc files contain time gaps, that assumption
    breaks. This subclass:

        1. __init__: loads the pre-computed list of valid window starts
           and rebuilds the sampler so its index space is exactly
           [0, len(valid_windows)) instead of [0, total_len).
        2. __len__: returns len(valid_windows).
        3. __getitem__: translates each "valid_window index" handed out by
           the sampler into the parent's expected global index
           (global_start_of_file + start_time_idx), then defers to the
           parent's logic via super().__getitem__.

    Everything else - worker, transforms, multistep rollout state,
    set_epoch / initialize_batch / batch_call_count - is reused verbatim.
    """

    def __init__(self, *args, valid_windows_path=None, expected_interval_h=6, **kwargs):
        """
        Args:
            valid_windows_path (str): Path to JSON file containing valid window
                starts produced by `check_multistep.py`. Format:
                [{"file_id": 0, "start_time_idx": 13}, ...]
                Each entry guarantees that
                  times[start_time_idx : start_time_idx + history_len + forecast_len + 1]
                are evenly spaced by `expected_interval_h` hours.
            expected_interval_h (int): expected timestep spacing in hours.
                Only used by the per-sample sanity check; must match the value
                used when `check_multistep.py` was run.
            All other args/kwargs are forwarded to ERA5_MultiStep_Batcher.
        """
        # Call parent __init__ first so all files are open and ERA5_indices
        # is populated.
        super().__init__(*args, **kwargs)

        if valid_windows_path is None:
            raise ValueError(
                "valid_windows_path is required. "
                "Run check_multistep.py first to generate this file."
            )

        with open(valid_windows_path, "r") as f:
            self.valid_windows = json.load(f)

        if len(self.valid_windows) == 0:
            raise RuntimeError(
                f"No valid windows in {valid_windows_path}. "
                "Check your data for excessive time gaps."
            )

        self.expected_interval_h = expected_interval_h

        # Sanity check: the json was generated for the same (history_len,
        # forecast_len) we're configured for. We can't read the json header
        # here (the stats file is separate), so just verify the first window
        # actually has enough room in its file for our configured horizon.
        first = self.valid_windows[0]
        n_times_first = len(self.all_files[first["file_id"]]["time"])
        needed = self.history_len + self.forecast_len + 1
        if first["start_time_idx"] + needed > n_times_first:
            raise ValueError(
                f"valid_windows_path={valid_windows_path} was generated for "
                f"a shorter horizon than (history_len={self.history_len}, "
                f"forecast_len={self.forecast_len}) requires (window_size="
                f"{needed}). Re-run check_multistep.py with the right "
                f"--history_len / --forecast_len."
            )

        # ------------------------------------------------------------------ #
        # Rebuild the sampler so its index space is the valid windows only.
        # The parent's sampler was constructed against len(self) == parent
        # __len__ (which is now overridden), so we replace it with one that
        # matches the new length. We re-use the same rank / world_size /
        # shuffle / seed as the parent.
        # ------------------------------------------------------------------ #
        self.sampler = DistributedSampler(
            self,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=True,
        )
        # Reproduce the parent's bootstrap so batch_indices is populated
        # before the trainer calls set_epoch.
        self.sampler.set_epoch(0)
        self.batch_indices = list(self.sampler)
        if len(self.batch_indices) < self.batch_size:
            logger.warning(
                f"Gap-aware: batch size ({self.batch_size}) > number of "
                f"valid-window indices on this rank ({len(self.batch_indices)})."
                f" Resetting batch size."
            )
            self.batch_size = len(self.batch_indices)

        logger.info(
            f"GapAwareMultiStepBatcher: loaded {len(self.valid_windows)} "
            f"valid windows from {valid_windows_path} "
            f"(history_len={self.history_len}, forecast_len={self.forecast_len})."
        )

    # ---------------------------------------------------------------------- #
    # Length now reflects the number of valid windows, not the raw timestep
    # count. The DistributedSampler reads len(self) internally.
    # ---------------------------------------------------------------------- #
    def __len__(self):
        return len(self.valid_windows)

    # ---------------------------------------------------------------------- #
    # Translate a valid-window index into the global index the parent class
    # expects, then defer to the parent's __getitem__.
    #
    # The parent ignores its argument and reads self.current_batch_indices
    # directly, so we patch current_batch_indices in place: each entry is
    # remapped from "valid_windows[k] index" to "global start index".
    # ---------------------------------------------------------------------- #
    def __getitem__(self, _):
        # Mirror the parent's reset-on-rollout-finished logic. Without this,
        # current_batch_indices would only get rebuilt at the next
        # initialize_batch call, and our remap below would re-translate the
        # same already-translated globals.
        if self.forecast_step_counts[0] == self.forecast_len + 1:
            self.initialize_batch()

        # current_batch_indices is set by initialize_batch() to whatever the
        # sampler handed out. Under this subclass those values are indices
        # INTO self.valid_windows, not global indices yet. Translate them
        # in-place so the parent's worker() call sees the correct global
        # index. Guard with a flag so we don't re-translate within the same
        # rollout (initialize_batch resets this flag).
        if not getattr(self, "_indices_translated", False):
            translated = []
            for vw_idx in self.current_batch_indices:
                w = self.valid_windows[int(vw_idx)]
                global_start = self.ERA5_indices[str(w["file_id"])][1]
                translated.append(global_start + w["start_time_idx"])
            self.current_batch_indices = translated
            self._indices_translated = True

        return super().__getitem__(_)

    # ---------------------------------------------------------------------- #
    # initialize_batch is the single point where current_batch_indices is
    # repopulated from the sampler output. Override it only to clear the
    # translation flag so __getitem__ knows it needs to translate again.
    # ---------------------------------------------------------------------- #
    def initialize_batch(self):
        super().initialize_batch()
        self._indices_translated = False