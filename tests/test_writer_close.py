"""The pipelined loop's shard writer when the run stops: a Ctrl-C (SIGINT) while close() writes the last output files,
or while it waits for room in a full queue, still lets the queued files be written and is raised afterwards; after a
loop error, that error is logged before the queued files are written and named in the error of the interrupted rank;
close() counts the file being written too.

CPU only. The first write waits until the test releases it, so files stay queued while close() waits.
"""

import contextlib
import logging
import os
import signal
import threading
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_pipelined_loop import _args, _Loader, _shard, fake_cuda  # noqa: E402, F401  (fake_cuda is a fixture)

from deeprm.inference import inference  # noqa: E402

ITEM = (np.zeros(2, np.float32), np.arange(2), np.zeros((2, 2), np.int64))  # preds, label_ids, read_ids


@pytest.fixture
def slow_disk(monkeypatch):
    """(started, release): the first write sets `started`, then waits for `release`."""
    started, release = threading.Event(), threading.Event()
    savez = np.savez_compressed

    def write(file, **arrays):
        started.set()
        assert release.wait(30), "the test never released the writer"
        savez(file, **arrays)

    monkeypatch.setattr(inference.np, "savez_compressed", write)
    yield started, release
    release.set()


@pytest.fixture
def sigint_raises():
    """SIGINT raises KeyboardInterrupt in the main thread, as in a terminal."""
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("SIGINT reaches only the main thread")
    old = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, old)


@contextlib.contextmanager
def _on_log(words, action):
    """Collects the inference logger's messages; starts action() in a new thread at the first one holding `words`."""
    messages = []

    class Handler(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())
            if words in messages[-1] and len([m for m in messages if words in m]) == 1:
                threading.Thread(target=action, daemon=True).start()

    handler = Handler(level=logging.DEBUG)
    inference.log.addHandler(handler)
    try:
        yield messages
    finally:
        inference.log.removeHandler(handler)


def _ctrl_c_then_release(release, delay=0.3):
    """A Ctrl-C to the main thread while close() waits, then the slow write is released."""

    def action():
        time.sleep(delay)  # close() logged its message just before it started to wait
        signal.pthread_kill(threading.main_thread().ident, signal.SIGINT)
        time.sleep(delay)
        release.set()

    return action


def test_a_ctrl_c_during_close_still_writes_the_queued_files(tmp_path, slow_disk, sigint_raises):
    started, release = slow_disk
    writer = inference._ShardWriter(output_logits=False)
    names = [f"inference_0_{i}.npz" for i in range(4)]
    for name in names:
        writer.put(str(tmp_path / name), *ITEM)
    assert started.wait(10)  # the first file is being written; the other three are queued
    with _on_log("Writing the last", _ctrl_c_then_release(release)) as messages:
        with pytest.raises(KeyboardInterrupt):
            writer.close()
    # all four files are complete when close() raises the Ctrl-C again; no .part file is left
    assert sorted(os.listdir(tmp_path)) == names
    assert any("Interrupted while writing the last output files" in m for m in messages), messages


def test_a_ctrl_c_while_close_waits_for_room_still_writes_the_files(tmp_path, slow_disk, sigint_raises):
    """One file is being written and WRITER_QUEUE wait, so close() first waits for room for its end marker. A Ctrl-C
    then is handled as one during the writes."""
    started, release = slow_disk
    writer = inference._ShardWriter(output_logits=False)
    names = [f"inference_0_{i}.npz" for i in range(inference.WRITER_QUEUE + 1)]
    writer.put(str(tmp_path / names[0]), *ITEM)
    assert started.wait(10)  # the first file is being written
    for name in names[1:]:
        writer.put(str(tmp_path / name), *ITEM)
    assert writer._queue.full()
    waiting, put = threading.Event(), writer._queue.put

    def put_and_tell(item, *args, **kwargs):
        if item is None:
            waiting.set()  # close() waits for room for its end marker
        return put(item, *args, **kwargs)

    writer._queue.put = put_and_tell
    ctrl_c_then_release = _ctrl_c_then_release(release)

    def action():
        assert waiting.wait(10)
        ctrl_c_then_release()

    threading.Thread(target=action, daemon=True).start()
    with _on_log("Interrupted while writing", lambda: None) as messages:
        with pytest.raises(KeyboardInterrupt):
            writer.close()
    assert sorted(os.listdir(tmp_path)) == sorted(names)  # all files complete, no .part file
    assert any("Interrupted while writing the last output files" in m for m in messages), messages


def test_a_ctrl_c_after_a_loop_error_keeps_the_error_and_the_queued_files(
    tmp_path,
    monkeypatch,
    fake_cuda,  # noqa: F811  (a fixture)
    slow_disk,
    sigint_raises,
):
    started, release = slow_disk

    class FailingModel:
        block_len = 9
        calls = 0

        def __call__(self, kmer, sig, seg_len, dwell_bq):
            FailingModel.calls += 1
            if FailingModel.calls == 6:  # shards 0-2 are with the writer by then (collected two batches behind)
                raise ValueError("A segment length is negative.")
            return kmer[:, :1].float()  # [B, 1]

    loader = _Loader([_shard(16, i) for i in range(8)])  # one 16-row batch per shard

    def gpu_rank(rank, args_dict):  # a GPU rank (num_gpu 1) that runs the pipelined loop
        inference._inference_loop_pipelined(_args(tmp_path), rank, torch.device("cpu"), FailingModel(), loader)

    monkeypatch.setattr(inference, "_inference_worker", gpu_rank)
    with _on_log("Writing the last", _ctrl_c_then_release(release)) as messages:
        with pytest.raises(RuntimeError, match="Rank 0 was interrupted") as interrupted:
            inference.inference_worker(0, {"num_gpu": 1})
    # the loop's own error is logged before the drain, and the interrupted rank's error names it
    expected = "Rank 0 stopped on ValueError('A segment length is negative.'); the output files already queued"
    assert any(expected in m for m in messages), messages
    assert "It had stopped on this error first: ValueError('A segment length is negative.')" in str(interrupted.value)
    assert isinstance(interrupted.value.__cause__, ValueError)
    assert sorted(os.listdir(tmp_path)) == ["inference_0_0.npz", "inference_0_1.npz", "inference_0_2.npz"]


def test_close_counts_the_file_being_written(tmp_path, slow_disk):
    started, release = slow_disk
    writer = inference._ShardWriter(output_logits=False)
    for i in range(3):
        writer.put(str(tmp_path / f"inference_0_{i}.npz"), *ITEM)
    assert started.wait(10)  # one file is being written, two are queued
    with _on_log("Writing the last", release.set) as messages:
        writer.close()
    assert "Writing the last 3 output file(s)." in messages, messages
    assert len(os.listdir(tmp_path)) == 3
