"""
Streaming dataset and dataloader for DeepRM inference.

This module provides an IterableDataset that reads preprocessed chunks
from a multiprocessing Queue instead of from disk, enabling streaming
from preprocessing directly into inference without intermediate files.
"""

from __future__ import annotations

from queue import Empty

from deeprm.inference.inference_dataloader import NanoporeDataLoader, collate_fn
from deeprm.utils import check_deps
from deeprm.utils.logging import get_logger

log = get_logger(__name__)
check_deps.check_torch_available()

from torch.utils.data import IterableDataset  # noqa: E402


class StreamingNanoporeIterator:
    """
    Iterator that reads preprocessed numpy-dict chunks from a multiprocessing Queue.

    Each chunk is a dict of numpy arrays (same format as the dicts produced by
    ``df_to_arrays`` in ``inference_preprocess_python.py``). Producers put
    ``None`` as a sentinel when they finish; the iterator stops after receiving
    one sentinel per producer. Error tuples ``("ERROR", pid, traceback_str)``
    are re-raised as RuntimeError.

    Args:
        data_queue (multiprocessing.Queue): Queue providing data chunks.
        num_producers (int): Number of preprocessing workers that will send data.
        timeout (int): Seconds to wait on an empty queue before raising TimeoutError.
    """

    def __init__(self, data_queue, num_producers, timeout=300):
        self.data_queue = data_queue
        self.num_producers = num_producers
        self.timeout = timeout
        self._sentinel_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                item = self.data_queue.get(timeout=self.timeout)
            except Empty:
                raise TimeoutError(
                    f"StreamingNanoporeIterator: no data received for {self.timeout}s. "
                    "Preprocessing may have stalled or crashed."
                )

            if item is None:
                self._sentinel_count += 1
                if self._sentinel_count >= self.num_producers:
                    raise StopIteration
                continue

            if isinstance(item, tuple) and len(item) >= 2 and item[0] == "ERROR":
                raise RuntimeError(f"Preprocessing worker {item[1]} failed: {item[2] if len(item) > 2 else 'unknown'}")

            return item


class StreamingNanoporeDataset(IterableDataset):
    """
    IterableDataset that yields preprocessed chunks from a multiprocessing Queue.

    Unlike the file-based NanoporeDataset, this dataset has indeterminate length
    and does not support resume or multi-worker DataLoader (num_workers must be 0).

    Args:
        data_queue (multiprocessing.Queue): Queue providing data chunks.
        num_producers (int): Number of preprocessing workers sending data.
        timeout (int): Seconds to wait for data before raising TimeoutError.
    """

    def __init__(self, data_queue, num_producers, timeout=300):
        super().__init__()
        self.data_queue = data_queue
        self.num_producers = num_producers
        self.timeout = timeout

    def __iter__(self):
        return StreamingNanoporeIterator(
            self.data_queue,
            self.num_producers,
            timeout=self.timeout,
        )

    def __len__(self):
        return 0


def load_stream_dataset(data_queue, num_producers, timeout=300):
    """
    Create a DataLoader that reads from a streaming queue.

    Args:
        data_queue (multiprocessing.Queue): Queue providing preprocessed chunks.
        num_producers (int): Number of preprocessing workers.
        timeout (int): Seconds to wait for data before raising TimeoutError.

    Returns:
        NanoporeDataLoader: DataLoader yielding GPU-ready dicts via collate_fn.
    """
    dataset = StreamingNanoporeDataset(data_queue, num_producers, timeout=timeout)
    dataloader = NanoporeDataLoader(
        dataset,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
        prefetch_factor=2,
    )
    return dataloader
