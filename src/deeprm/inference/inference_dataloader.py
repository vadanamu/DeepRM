"""Data loader for the .npz shards that deeprm call prep writes. One DataLoader item is one whole shard.

Shards are split across ranks (one per GPU), then across DataLoader workers; each worker reads several shards at once
in a thread pool. deeprm call run cuts each shard into model calls of at most ``--batch`` rows. Shapes use the
letters of deeprm.model.deeprm_v12 (M = 21, L = 17, R = 1224), except that B is the rows of one shard.
"""

import glob
import math
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from deeprm.utils.logging import get_logger

log = get_logger(__name__)

# Attempts per shard before it is skipped; the reader waits 1 s, 2 s, then 3 s after the failed attempts.
READ_RETRIES = 3
# The float arrays of a shard (see _read_df); the other arrays hold integers, which are always finite.
FLOAT_KEYS = ("signal_token", "dwell_motor_token", "dwell_pore_token")
# Set only for a shard that has rows with a NaN or infinite value: [B] bool, True for those rows.
NONFINITE_KEY = "nonfinite_rows"


def zero_nonfinite_rows(data, path):
    """Zeros the float values of every row that holds a NaN or infinite value, and marks those rows.

    DeepRM's prep never writes such a value; a damaged or hand-made shard can. In the fa4 and fast engines one such
    row would make other rows of its model call NaN too: those engines add up the centre tokens of all rows in one
    running sum, and fa4 reads keys and values in 128-row tiles that reach into the neighbouring rows. With its values
    zeroed the row is computed like any other, and deeprm call run writes NaN as its prediction. A shard without such
    rows is returned unchanged (no new key, no copy). Returns the number of marked rows.

    The values are checked as float32, the type collate_fn gives the model, so a float64 value beyond the float32
    maximum (about 3.4e38), which the cast turns into Inf, counts as infinite too.
    """
    n_rows = data["label_id"].shape[0] if data["label_id"].ndim else 0
    # A malformed array is left out and left as it is: the model refuses it later, as without this check.
    arrays = [
        data[key]
        for key in FLOAT_KEYS
        if data[key].ndim >= 2 and data[key].shape[0] == n_rows and np.issubdtype(data[key].dtype, np.floating)
    ]
    bad = np.zeros(n_rows, dtype=bool)
    for values in arrays:
        # copy=False: a float32 array (all that prep writes) is checked as it is, without a copy
        with np.errstate(over="ignore"):  # the overflow to Inf is what is looked for here, not a mistake
            as_float32 = values.astype(np.float32, copy=False)
        bad |= ~np.isfinite(as_float32).all(axis=tuple(range(1, values.ndim)))  # [B]
    n_bad = int(bad.sum())
    if n_bad == 0:
        return 0
    for values in arrays:
        values[bad] = 0  # npz[key] made a new array, so the file is not changed
    data[NONFINITE_KEY] = bad
    # One warning per shard: the reader knows the file, and a damaged input usually affects few shards.
    log.warning(
        f"{path}: {n_bad} of {n_rows} rows hold a NaN or infinite signal or dwell value, which DeepRM's prep never "
        "writes; the file may be damaged. Their predictions are NaN (the pileup skips them); the other rows are not "
        "affected, apart from float rounding."
    )
    return n_bad


# The arrays every shard holds, under the names deeprm call prep gives them.
SHARD_KEYS = (
    "read_id",
    "label_id",
    "segment_len_arr",
    "signal_token",
    "kmer_token",
    "dwell_motor_token",
    "dwell_pore_token",
    "bq_token",
)

# The end of the warning for a skipped shard, whether it could not be read or is malformed.
SKIPPED_SHARD_ADVICE = (
    "Its reads are missing from the results. To add them, re-create the shard under the same name and run again with "
    "--resume: the run then starts again from that shard. Do not remove the shard: --resume refuses an input folder "
    "whose files changed."
)


def shard_problem(data, cb_len=21, kmer_len=5, sampling=6, sig_window=5, max_seq_len=None):
    """Returns why the model cannot take a shard's arrays (a short phrase), or None if it can.

    Both preprocessings (Python and C++) write one row per site in every array, with fixed widths: L = cb_len -
    kmer_len + 1 columns in segment_len_arr, the two dwell tokens and bq_token, M = cb_len in kmer_token, 2 in read_id,
    and (--max-token-len + kmer_len - 1) * sampling samples in signal_token, which the model cuts into exactly
    --max-token-len signal windows. They also drop every site whose segment lengths add up to more than
    --max-token-len, so no row they write is longer than its signal. A shard that breaks a rule here is damaged, was
    not written by deeprm call prep, or was written by it with options that do not fit the model (another --cb-len or
    --kmer-len, or a --max-token-len over max_seq_len). The model would stop the whole run on it; it is skipped
    instead, like an unreadable shard. The run stops at the start, before any shard is read, only if the array headers
    (shapes and dtypes) of the folder's first, middle and last shard break the same rule (see folder_problem). Shards
    that break different rules, or a rule on the values (a negative length, a row longer than its signal), are
    skipped one at a time.

    signal_token's width is not fixed: it follows the prep's --max-token-len. The windows are counted as the model
    counts them, (R - window) // stride + 1: the loader's sampling is the model's signal_stride, sig_window its
    kmer_size.
    """
    n_bases = cb_len - kmer_len + 1  # L
    widths = {  # None: any width
        "read_id": 2,
        "segment_len": n_bases,
        "signal_token": None,
        "kmer_token": cb_len,
        "dwell_motor_token": n_bases,
        "dwell_pore_token": n_bases,
        "bq_token": n_bases,
    }
    for key, value in data.items():
        name = "segment_len_arr" if key == "segment_len" else key  # the name in the file
        # ids, lengths and ASCII codes must be integers; signal, dwell and base quality any number type
        kinds = "iu" if key in ("read_id", "label_id", "segment_len", "kmer_token") else "biuf"
        if value.dtype.kind not in kinds:
            return f"{name} has dtype {value.dtype}, which the model cannot take"
    if data["label_id"].ndim != 1:
        return f"label_id has shape {data['label_id'].shape}; it must have one value per row"
    n_rows = data["label_id"].shape[0]
    for key, width in widths.items():
        name = "segment_len_arr" if key == "segment_len" else key
        shape = data[key].shape
        if len(shape) != 2:
            return f"{name} has shape {shape}; it must have 2 dimensions"
        if shape[0] != n_rows:
            return f"{name} has {shape[0]} rows, but label_id has {n_rows}"
        if width is not None and shape[1] != width:
            return f"{name} has {shape[1]} columns; this model needs {width}"
    n_samples = data["signal_token"].shape[1]  # R
    window = sampling * sig_window
    if n_samples < window:
        return f"signal_token has {n_samples} samples per row; the model needs at least {window}"
    n_windows = (n_samples - window) // sampling + 1  # S
    if max_seq_len is not None and n_windows > max_seq_len:
        return f"signal_token gives {n_windows} signal windows per row; this model takes at most {max_seq_len}"
    seg = data["segment_len"]
    if (seg < 0).any():
        return "segment_len_arr holds a negative length"
    too_long = int((seg.sum(axis=1, dtype=np.int64) > n_windows).sum())
    if too_long:
        return (
            f"in {too_long} row(s), segment_len_arr adds up to more than the {n_windows} signal windows of "
            "signal_token"
        )
    return None


def _header_problem(path, cb_len, kmer_len, sampling, sig_window, max_seq_len):
    """shard_problem for one shard, read from its array headers only (shapes and dtypes; no data is decompressed).

    None if the shard passes, or if its headers cannot be read this way: the reader then reads the file as usual.
    """
    readers = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
    data = {}
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            missing = [key for key in SHARD_KEYS if f"{key}.npy" not in names]
            if missing:
                return f"it lacks the array(s) {', '.join(missing)}"
            for key in SHARD_KEYS:
                with archive.open(f"{key}.npy") as fh:
                    read_header = readers.get(np.lib.format.read_magic(fh))
                    if read_header is None:
                        return None
                    shape, _, dtype = read_header(fh)
                # zeros with the file's shape and dtype that take no memory; zeros pass the rules on values
                data["segment_len" if key == "segment_len_arr" else key] = np.broadcast_to(np.zeros((), dtype), shape)
    except Exception:  # noqa: BLE001 -- not readable this way: left to the reader
        return None
    return shard_problem(data, cb_len, kmer_len, sampling, sig_window, max_seq_len=max_seq_len)


def folder_problem(file_paths, cb_len=21, kmer_len=5, sampling=6, sig_window=5, max_seq_len=None):
    """Returns (shard paths, reason) if the first, middle and last shard all give the same reason (the same text,
    numbers included) from the array-header checks of shard_problem; else None.

    One shard that does not fit the model is skipped with a warning. But a folder that prep wrote with other
    geometry options fits in none of its shards: the run would print one warning per shard and stop only after
    reading them all. Prep writes every shard of a folder with the same widths, so three shards with the same
    problem stand for the folder, and the run can stop at the start with one error. Only the array headers are read,
    before the data of any shard.
    """
    if not file_paths:
        return None
    picked = [file_paths[i] for i in sorted({0, len(file_paths) // 2, len(file_paths) - 1})]
    problems = {_header_problem(path, cb_len, kmer_len, sampling, sig_window, max_seq_len) for path in picked}
    if len(problems) == 1 and None not in problems:
        return picked, problems.pop()
    return None


class NanoporeDatasetIterator:
    """Returns the shards of file_paths one at a time, in order: a dict of numpy arrays, or None if unreadable or
    malformed.

    A thread pool reads the next max_workers shards as one group, at the same time. The next group starts only after
    the last shard of the current group is returned. cb_len, kmer_len, sampling, sig_window and max_seq_len only
    check each shard's widths (shard_problem): the prep already cut every array to its final width. cb_lr_pad and
    trim are stored but unused.
    """

    def __init__(self, file_paths, cb_len=21, kmer_len=5, sampling=6, sig_window=5, max_workers=4, max_seq_len=None):
        self.file_paths = file_paths
        self.cb_len = cb_len
        self.kmer_len = kmer_len
        self.sampling = sampling
        self.sig_window = sig_window
        self.max_seq_len = max_seq_len
        self.cb_lr_pad = (cb_len - kmer_len) // 2
        self.trim = kmer_len // 2
        self.max_workers = max(1, int(max_workers))
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._file_index = 0  # first shard not yet submitted to the pool
        self._futures = []  # the current group's shards not yet returned, in file order
        self._closed = False

    def _read_df(self, path):
        """Reads one shard into a dict of numpy arrays, or returns None if it cannot be read or is malformed (see
        shard_problem). Runs in a reader thread.

        The arrays as the prep writes them::

            read_id            [B, 2]  int64    the read's 16-byte UUID as two int64 values
            label_id           [B]     int64    (reference index * label_div + position + 1) * strand (+1 / -1);
                                                label_div is 10**9 by default
            segment_len        [B, L]  uint16   6-sample steps (= signal windows) of each base; key segment_len_arr
            signal_token       [B, R]  float32  normalised signal, zero-padded at the end
            kmer_token         [B, M]  uint8    ASCII codes of the bases
            dwell_motor_token  [B, L]  float32  scaled log dwell of the base 10 places later
            dwell_pore_token   [B, L]  float32  scaled log dwell of the base itself
            bq_token           [B, L]  uint8    base quality, clipped to 0..60
        """
        # An unreadable shard is retried, then returned as None, and so is a malformed one, without retries: it keeps
        # its place, so the output files stay numbered like the input shards.
        last_exc = None
        for attempt in range(READ_RETRIES):
            try:
                # npz[key] decompresses into a new array, so the dict stays valid after the file is closed.
                with np.load(path) as npz:
                    # A missing array is not retried: the file was read, it is just not a whole shard.
                    missing = [key for key in SHARD_KEYS if key not in npz.files]
                    if missing:
                        break
                    data = {
                        "read_id": npz["read_id"],
                        "label_id": npz["label_id"],
                        "segment_len": npz["segment_len_arr"],
                        "signal_token": npz["signal_token"],
                        "kmer_token": npz["kmer_token"],
                        "dwell_motor_token": npz["dwell_motor_token"],
                        "dwell_pore_token": npz["dwell_pore_token"],
                        "bq_token": npz["bq_token"],
                    }
                break
            except Exception as exc:
                # Any error counts as unreadable: a damaged or partly written file.
                last_exc = exc
                log.warning(f"Failed to read {path} (attempt {attempt + 1}/{READ_RETRIES}): {exc!r}")
                time.sleep(attempt + 1)
        else:
            log.warning(
                f"Skipping input shard {path}: it could not be read in {READ_RETRIES} attempts ({last_exc!r}). "
                f"{SKIPPED_SHARD_ADVICE}"
            )
            return None
        if missing:
            problem = f"it lacks the array(s) {', '.join(missing)}"
        else:
            problem = shard_problem(
                data, self.cb_len, self.kmer_len, self.sampling, self.sig_window, max_seq_len=self.max_seq_len
            )
        if problem:
            log.warning(f"Skipping input shard {path}: {problem}. {SKIPPED_SHARD_ADVICE}")
            return None
        # Here, in the reader thread, so the GPU loop does not pay for the check.
        zero_nonfinite_rows(data, path)
        return data

    def _fill_futures(self):
        # __next__ returns the shards in the order they were submitted, so the file order is kept.
        if self._closed or self._futures or self._file_index >= len(self.file_paths):
            return
        end_index = min(self._file_index + self.max_workers, len(self.file_paths))
        paths_batch = self.file_paths[self._file_index : end_index]
        self._futures = [self.executor.submit(self._read_df, p) for p in paths_batch]
        self._file_index = end_index

    def close(self):
        if not self._closed:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self._closed = True
            self._futures = []

    def __iter__(self):
        """Returns self."""
        return self

    def __next__(self):
        """Returns the next shard's arrays, or None for a shard that could not be read or is malformed."""
        self._fill_futures()
        if not self._futures:
            # Every shard has been returned.
            self.close()
            raise StopIteration

        future = self._futures.pop(0)  # the oldest shard of the group
        return future.result()

    def __del__(self):
        # Also shuts the threads down when the iterator is dropped before the end.
        self.close()


class NanoporeDataset(IterableDataset):
    """Iterable dataset over the shards of one rank, in sorted order. Each item is one whole shard (or None).

    Args:
        num_files_read_once (int): Reader threads per worker process (deeprm call run passes ``--prefetch``).
        resume_from (int): Shards of this rank to skip, not shards of the whole folder.
        dataloader_workers (int): DataLoader worker count; splits the shards when iterated outside a worker process.
        max_seq_len (int): Most signal windows a row may have (the model's max_seq_len); None does not check it.

    seed, epoch, num_shard and skip are stored but unused: inference reads every shard once, in order.
    """

    def __init__(
        self,
        data_path,
        rank,
        num_replicas,
        seed=0,
        num_files_read_once=1000,
        cb_len=21,
        kmer_len=5,
        sampling=6,
        sig_window=5,
        resume_from=0,
        dataloader_workers=0,
        max_seq_len=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.rank = rank
        self.num_replicas = num_replicas
        # Sorted: every rank sees the same list, and every run numbers the shards the same way.
        self.file_paths = sorted(glob.glob(os.path.join(self.data_path, "*.npz")))
        self.epoch = 0
        self.seed = seed
        self.num_shard = math.ceil(len(self.file_paths) / max(1, num_replicas)) if self.file_paths else 0
        self.num_files_read_once = max(1, int(num_files_read_once))
        self.cb_len = cb_len
        self.kmer_len = kmer_len
        self.sampling = sampling
        self.sig_window = sig_window
        self.resume_from = max(0, int(resume_from))
        self.skip = 0
        self.dataloader_workers = max(0, int(dataloader_workers))
        self.max_seq_len = max_seq_len

    def _get_partition(self, worker_info=None):
        """Returns the shard paths of this rank and DataLoader worker: strided across ranks, then across workers.

        resume_from counts this rank's shards, so it is applied before the worker split. Example: of 10 shards and 2
        ranks, rank 0 takes [0, 2, 4, 6, 8], and resume_from = 1 leaves [2, 4, 6, 8]; 3 workers read [2, 8], [4], [6].
        The DataLoader takes one item from each worker in turn and skips a worker that has run out, so they arrive as
        [2, 4, 6, 8]. This holds for any shard and worker count: the strided split never gives an earlier worker fewer
        shards. inference.py numbers its output files in this order, which --resume relies on.
        """
        rank_file_paths = self.file_paths[self.rank :: max(1, self.num_replicas)]

        if self.resume_from:
            rank_file_paths = rank_file_paths[self.resume_from :]

        if worker_info is None:
            # Not in a worker process: load_dataset passed dataloader_workers = 0, so this process takes every shard.
            # Iterated directly with dataloader_workers > 0, the dataset returns worker 0's share only.
            worker_id = 0
            num_workers = max(1, self.dataloader_workers)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # A worker beyond the shard count gets [].
        return rank_file_paths[worker_id::num_workers]

    def __iter__(self):
        """Returns an iterator over this worker's shards. Called once per worker, so each has its own thread pool."""
        worker_info = torch.utils.data.get_worker_info()
        file_paths = self._get_partition(worker_info)

        # No more reader threads than this worker has shards.
        return NanoporeDatasetIterator(
            file_paths,
            cb_len=self.cb_len,
            kmer_len=self.kmer_len,
            sampling=self.sampling,
            sig_window=self.sig_window,
            max_workers=min(self.num_files_read_once, max(1, len(file_paths))) if file_paths else 1,
            max_seq_len=self.max_seq_len,
        )

    def __len__(self):
        """Returns this rank's shard count after resume_from, unreadable shards included."""
        # Not one worker's share: PyTorch adds up the items of all workers and warns if the total passes len().
        rank_file_paths = self.file_paths[self.rank :: max(1, self.num_replicas)]
        if self.resume_from:
            rank_file_paths = rank_file_paths[self.resume_from :]
        return len(rank_file_paths)


class NanoporeDataLoader(DataLoader):
    """DataLoader with automatic batching off: each item, one whole shard, goes to collate_fn on its own.

    drop_last must be False: with batch_size=None PyTorch raises ValueError for True. prefetch_factor counts shards.
    """

    def __init__(self, dataset: NanoporeDataset, num_workers, pin_memory, drop_last, collate_fn, prefetch_factor):
        # PyTorch allows no shuffle or sampler with an IterableDataset; the dataset keeps the file order itself.
        kwargs = dict(
            dataset=dataset,
            batch_size=None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            shuffle=False,
            sampler=None,
            collate_fn=collate_fn,
        )
        # PyTorch accepts prefetch_factor only with worker processes.
        if num_workers > 0:
            kwargs["prefetch_factor"] = prefetch_factor
            kwargs["persistent_workers"] = False
        super().__init__(**kwargs)


def load_dataset(
    data_path,
    rank,
    num_replicas,
    num_files_read_once=1,
    prefetch_factor=2,
    worker=16,
    cb_len=21,
    kmer_len=5,
    sampling=6,
    sig_window=5,
    resume_from=0,
    pin_memory=True,
    max_seq_len=None,
):
    """Returns a NanoporeDataLoader over this rank's shards in data_path, with `worker` worker processes.

    deeprm call run passes ``--prefetch`` as both num_files_read_once and prefetch_factor: each worker reads up to
    ``--prefetch`` shards at once and keeps up to ``--prefetch`` shards ready. pin_memory is ignored when worker is 0.
    The geometry (cb_len to sig_window) and max_seq_len come from the model; a shard that does not fit them is
    skipped with a warning (see shard_problem). Raises ValueError, before any shard is read, for a folder whose first,
    middle and last shard all fail in the same way (see folder_problem).
    """
    dataset = NanoporeDataset(
        data_path,
        rank,
        num_replicas,
        num_files_read_once=num_files_read_once,
        cb_len=cb_len,
        kmer_len=kmer_len,
        sampling=sampling,
        sig_window=sig_window,
        resume_from=resume_from,
        dataloader_workers=worker,
        max_seq_len=max_seq_len,
    )
    found = folder_problem(dataset.file_paths, cb_len, kmer_len, sampling, sig_window, max_seq_len)
    if found:
        paths, problem = found
        limit = f" and a --max-token-len of at most {max_seq_len}" if max_seq_len is not None else ""
        raise ValueError(
            f"The input shards do not fit this model: {problem} (in {', '.join(os.path.basename(p) for p in paths)}; "
            f"the run checks the first, middle and last shard of {data_path} before it starts). Check that --input is "
            f"a folder that deeprm call prep wrote with --cb-len {cb_len}, --kmer-len {kmer_len}{limit}, as this "
            "model needs."
        )
    # With worker = 0 PyTorch would pin in the main thread, so pinning is off; with workers a background thread pins.
    dataloader = NanoporeDataLoader(
        dataset,
        num_workers=worker,
        pin_memory=pin_memory and worker > 0,
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=max(1, int(prefetch_factor)),
    )
    return dataloader


def collate_fn(batch):
    """Turns one shard's arrays into a dict of tensors; None stays None. With batch_size=None, `batch` is one shard."""
    if batch is None:
        return None
    # read_id, label_id and signal_token already have their dtype, so as_tensor shares their memory (no copy).
    source = {}
    source["read_id"] = torch.as_tensor(batch["read_id"])  # [B, 2] int64
    source["label_id"] = torch.as_tensor(batch["label_id"])  # [B] int64
    source["segment_len"] = torch.as_tensor(batch["segment_len"], dtype=torch.int32)  # [B, L] uint16 -> int32
    source["signal_token"] = torch.as_tensor(batch["signal_token"], dtype=torch.float32)  # [B, R] float32
    source["kmer_token"] = torch.as_tensor(batch["kmer_token"], dtype=torch.int32)  # [B, M] uint8 -> int32
    # Three [B, L] arrays -> [B, L, 3]. The v1.2 model does not read it (inference.to_device passes None).
    source["dwell_bq_token"] = torch.as_tensor(
        np.stack(
            (
                batch["dwell_motor_token"],
                batch["dwell_pore_token"],
                batch["bq_token"],
            ),
            axis=-1,
        ),
        dtype=torch.float32,
    )
    if NONFINITE_KEY in batch:
        source[NONFINITE_KEY] = torch.as_tensor(batch[NONFINITE_KEY])  # [B] bool; see zero_nonfinite_rows
    return source
