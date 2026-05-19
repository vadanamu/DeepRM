"""
Dataloader for Nanopore Dataset from NPZ Files

This module provides an iterator and dataset class for loading
Nanopore data from NPZ files. It supports parallel reading of files
and batching of data for efficient processing in PyTorch.
"""

import glob
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from deeprm.utils import check_deps
from deeprm.utils.logging import get_logger

check_deps.check_torch_available()

import torch
from torch.utils.data import DataLoader, IterableDataset

log = get_logger(__name__)
check_deps.check_torch_available()


class NanoporeDatasetIterator:
    """
    Iterator for loading Nanopore dataset from NPZ files.

    Args:
        file_paths (list): List of file paths to NPZ files.
        cb_len (int): Context block length.
        kmer_len (int): K-mer length.
        sampling (int): Sampling rate.
        sig_window (int): Signal window size.
    """

    def __init__(self, file_paths, cb_len=21, kmer_len=5, sampling=6, sig_window=5, max_workers=4):
        self.file_paths = file_paths
        self.cb_len = cb_len
        self.kmer_len = kmer_len
        self.sampling = sampling
        self.sig_window = sig_window
        self.cb_lr_pad = (cb_len - kmer_len) // 2
        self.trim = kmer_len // 2
        self.max_workers = max(1, int(max_workers))
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._file_index = 0
        self._futures = []
        self._closed = False

    def _read_df(self, path):
        """
        Reads a single NPZ file and returns the data as a dictionary.

        Args:
            path (str): Path to the NPZ file.

        Returns:
            dict: Dictionary containing the data from the NPZ file.
        """
        try:
            with np.load(path) as npz:
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
        except Exception:
            log.warning(f"Failed to read {path}, skipping.")
            return None
        return data

    def _fill_futures(self):
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
        """
        Returns the iterator object itself.

        Returns:
            NanoporeDatasetIterator: The iterator object.
        """
        return self

    def __next__(self):
        """
        Returns the next data from the iterator.

        Returns:
            dict: Dictionary containing the data from the next NPZ file.

        Raises:
            StopIteration: If there are no more files to read.
        """
        self._fill_futures()
        if not self._futures:
            self.close()
            raise StopIteration

        future = self._futures.pop(0)
        data = future.result()
        if data is None:
            return self.__next__()
        return data

    def __del__(self):
        self.close()


class NanoporeDataset(IterableDataset):
    """
    Iterable dataset for loading Nanopore data from NPZ files.

    Args:
        data_path (str): Path to the directory containing NPZ files.
        rank (int): Rank of the current process.
        num_replicas (int): Number of replicas.
        seed (int): Random seed.
        num_files_read_once (int): Number of files to read at once.
        cb_len (int): Context block length.
        kmer_len (int): K-mer length.
        sampling (int): Sampling rate.
        sig_window (int): Signal window size.
        resume_from (int): Number of files to skip from the start.
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
    ):
        super().__init__()
        self.data_path = data_path
        self.rank = rank
        self.num_replicas = num_replicas
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

    def _get_partition(self, worker_info=None):
        """
        Compute the exact file list partition for this rank / worker replica.

        Important:
        - resume_from is applied BEFORE worker subdivision, because it refers to
          the rank-local shard index used by inference_worker.
        - The same partitioning logic is used by both __iter__() and __len__(),
          so the reported dataset length matches the actual number of yielded
          samples under multiprocessing.
        """
        # First split files across GPU/process replicas (rank-local view).
        rank_file_paths = self.file_paths[self.rank :: max(1, self.num_replicas)]

        # Resume counts rank-local shards, so apply it here.
        if self.resume_from:
            rank_file_paths = rank_file_paths[self.resume_from :]

        # Then split the rank-local view across DataLoader workers.
        if worker_info is None:
            worker_id = 0
            num_workers = max(1, self.dataloader_workers)
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        return rank_file_paths[worker_id::num_workers]

    def __iter__(self):
        """
        Returns an iterator for the dataset.

        Returns:
            NanoporeDatasetIterator: Iterator for the dataset.
        """
        worker_info = torch.utils.data.get_worker_info()
        file_paths = self._get_partition(worker_info)

        return NanoporeDatasetIterator(
            file_paths,
            cb_len=self.cb_len,
            kmer_len=self.kmer_len,
            sampling=self.sampling,
            sig_window=self.sig_window,
            max_workers=min(self.num_files_read_once, max(1, len(file_paths))) if file_paths else 1,
        )

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of shards in the dataset.
        """
        # Total samples yielded across all DataLoader workers for this rank.
        # Must NOT be subdivided by worker count: each worker iterates its own
        # slice and DataLoader._num_yielded sums them, so len() is compared
        # against the rank-local total, not a per-worker share.
        rank_file_paths = self.file_paths[self.rank :: max(1, self.num_replicas)]
        if self.resume_from:
            rank_file_paths = rank_file_paths[self.resume_from :]
        return len(rank_file_paths)


class NanoporeDataLoader(DataLoader):
    """
    DataLoader for loading Nanopore data.

    Args:
        dataset (NanoporeDataset): The dataset to load data from.
        num_workers (int): Number of worker processes.
        pin_memory (bool): Whether to pin memory.
        drop_last (bool): Whether to drop the last incomplete batch.
        collate_fn (typing.Callable): Function to collate data into batches.
        prefetch_factor (int): Number of batches to prefetch.
    """

    def __init__(self, dataset: NanoporeDataset, num_workers, pin_memory, drop_last, collate_fn, prefetch_factor):
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
):
    """
    Loads the Nanopore dataset using DataLoader.

    Args:
        data_path (str): Path to the directory containing NPZ files.
        batch_size (int): Batch size for loading data.
        rank (int): Rank of the current process.
        num_replicas (int): Number of replicas.
        pad_to (int): Padding length for sequences.
        bq_clip (int): Base quality clipping value.
        num_files_read_once (int): Number of files to read at once.
        prefetch_factor (int): Number of batches to prefetch.
        worker (int): Number of worker processes.
        cb_len (int): Context block length.
        kmer_len (int): K-mer length.
        sampling (int): Sampling rate.
        sig_window (int): Signal window size.
        resume_from (int): Number of files to skip from the start.

    Returns:
        NanoporeDataLoader: DataLoader for loading the dataset.
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
    )
    dataloader = NanoporeDataLoader(
        dataset,
        num_workers=worker,
        pin_memory=worker > 0,
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=max(1, int(prefetch_factor)),
    )
    return dataloader


def collate_fn(batch):
    """
    Collate function to process a batch of data from the Nanopore dataset.

    Args:
        batch (list): List of dictionaries containing data from the dataset.

    Returns:
        dict: Dictionary containing processed data ready for model input.
    """
    source = {}
    source["read_id"] = torch.as_tensor(batch["read_id"])
    source["label_id"] = torch.as_tensor(batch["label_id"])
    source["segment_len"] = torch.as_tensor(batch["segment_len"], dtype=torch.int32)
    source["signal_token"] = torch.as_tensor(batch["signal_token"], dtype=torch.float32)
    source["kmer_token"] = torch.as_tensor(batch["kmer_token"], dtype=torch.int32)
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
    return source
