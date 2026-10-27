"""The pipelined GPU inference loop, its host-side token counts and its shard writer, run on the CPU: the CUDA streams
and events are replaced by no-ops, so the loop's own logic (ordering, numbering, files, errors) is what is tested."""

import collections
import contextlib
import os
import threading

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_v12_model import _engine, _synthetic_inputs, save_dict  # noqa: E402, F401  (save_dict is a fixture)

from deeprm.inference import inference  # noqa: E402


class _FakeStream:
    def __init__(self, *args, **kwargs):
        pass

    def wait_stream(self, other):
        pass


class _FakeEvent:
    def record(self, stream=None):
        pass

    def synchronize(self):
        pass


@pytest.fixture
def fake_cuda(monkeypatch):
    """No-op CUDA streams and events, so the pipelined loop runs on the CPU."""
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _FakeStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda self, stream: None)


def _shard(n, seed):
    kmer, signal, seg = _synthetic_inputs(n, seed)
    return {
        "kmer_token": kmer,
        "signal_token": signal,
        "segment_len": seg,
        "label_id": torch.arange(n, dtype=torch.int64) + 1000 * seed,
        "read_id": torch.stack([torch.arange(n), torch.full((n,), seed)], 1).to(torch.int64),
    }


class _Loader(list):
    """A list of shards (None = a shard that could not be read), as the data loader yields them."""


def _args(out, batch=16):
    return {"batch": batch, "output": str(out), "output_id": None, "flush": 100, "output_logits": True,
            "dwell_bq_channels": None}


def _files(folder):
    found = {}
    for name in sorted(os.listdir(folder)):
        assert not name.endswith(".part"), name
        with np.load(os.path.join(folder, name)) as f:
            found[name] = {key: f[key] for key in ("label_id", "read_id", "pred")}
    return found


@pytest.mark.parametrize("engine", ["fa4", "fast", "eager"])
def test_pipelined_loop_writes_what_the_sync_loop_writes(save_dict, fake_cuda, tmp_path, engine):  # noqa: F811
    import fa4_cpu_standin

    fa4_cpu_standin.enable()
    model = _engine(save_dict, engine)
    loader = _Loader([_shard(40, 1), None, _shard(33, 2), _shard(0, 3), _shard(1, 4), _shard(16, 5)])
    sync_dir, pipe_dir = tmp_path / "sync", tmp_path / "pipe"
    sync_dir.mkdir(), pipe_dir.mkdir()
    cpu = torch.device("cpu")
    inference._inference_loop_sync(_args(sync_dir), 0, cpu, model, loader, start_index=7)
    inference._inference_loop_pipelined(_args(pipe_dir), 0, cpu, model, loader, start_index=7)
    sync, pipe = _files(sync_dir), _files(pipe_dir)
    # shard 8 could not be read and shard 10 is empty: no file for either; numbering starts at the resume point 7
    assert list(sync) == ["inference_0_11.npz", "inference_0_12.npz", "inference_0_7.npz", "inference_0_9.npz"]
    assert list(pipe) == list(sync)
    for name in sync:
        for key in ("label_id", "read_id"):
            assert np.array_equal(pipe[name][key], sync[name][key])
        assert pipe[name]["pred"].dtype == np.float32
        assert np.array_equal(pipe[name]["pred"].view(np.uint32), sync[name]["pred"].view(np.uint32))


def test_host_counts_give_the_same_output_as_device_counts(save_dict):  # noqa: F811
    import fa4_cpu_standin

    fa4_cpu_standin.enable()
    model = _engine(save_dict, "fa4")
    centre = model.block_len // 2
    for n, seed in ((64, 3), (3, 4), (1, 5)):  # one row: the engine runs it twice, so the counts are doubled
        kmer, signal, seg = _synthetic_inputs(n, seed)
        if n == 64:
            seg[0] = 0  # a row with no signal
            seg[1, centre] = 0  # a centre base without windows
        with torch.inference_mode():
            device_counts = model(kmer, signal, seg, None)
            host_counts = model(kmer, signal, seg, None, counts=inference.batch_counts(seg, centre))
        assert np.array_equal(device_counts.numpy().view(np.uint32), host_counts.numpy().view(np.uint32))
    seg[0, 0] = -1  # a negative length is refused on both paths
    for counts in (None, inference.batch_counts(seg, centre)):
        with pytest.raises(ValueError, match="negative"), torch.inference_mode():
            model(kmer, signal, seg, None, counts=counts)
    # each host count is the engine's own: the centre count is clamped at 0 on both paths
    seg[0, centre] = -3
    row_len = seg.sum(1)
    engine = (row_len.sum(), seg[:, centre].long().clamp(min=0).sum(), row_len.max(), (seg < 0).any())  # Fa4Engine's
    assert inference.batch_counts(seg, centre) == (int(engine[0]), int(engine[1]), int(engine[2]), bool(engine[3]))


def _run_with_timeout(fn, seconds=30):
    result = {}

    def target():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), "the call hung"
    return result.get("error")


def test_shard_writer_writes_complete_files(tmp_path):
    writer = inference._ShardWriter(output_logits=True)
    preds = np.array([0.0, 2.0, np.nan], dtype=np.float32)
    writer.put(str(tmp_path / "inference_0_0.npz"), preds, np.arange(3), np.zeros((3, 2), np.int64))
    writer.close()
    assert sorted(os.listdir(tmp_path)) == ["inference_0_0.npz"]
    with np.load(tmp_path / "inference_0_0.npz") as f:
        assert np.array_equal(f["pred"], inference.sigmoid_np(preds), equal_nan=True)


@pytest.mark.parametrize("exc_type", [OSError, KeyboardInterrupt])
def test_shard_writer_reports_a_failed_write_and_never_hangs(tmp_path, monkeypatch, exc_type):
    def failing_write(*args, **kwargs):
        raise exc_type("disk full")

    monkeypatch.setattr(inference.np, "savez_compressed", failing_write)
    writer = inference._ShardWriter(output_logits=False)
    item = (np.zeros(2, np.float32), np.arange(2), np.zeros((2, 2), np.int64))

    def fill():
        for i in range(inference.WRITER_QUEUE + 4):  # more than the queue holds, after the writer stopped
            writer.put(str(tmp_path / f"inference_0_{i}.npz"), *item)

    error = _run_with_timeout(fill)
    assert isinstance(error, RuntimeError) and "disk full" in repr(error.__cause__)
    assert _run_with_timeout(writer.close) is None  # reported once: close() neither hangs nor raises it again
    assert not any(name.endswith(".npz") for name in os.listdir(tmp_path))


def test_collected_predictions_do_not_keep_their_host_buffers(fake_cuda):
    """float32 predictions (the eager engine) are copied, so each batch's pinned buffer is freed at once."""

    class Writer:
        def check(self):
            pass

        def put(self, out_path, preds, label_ids, read_ids):
            self.preds = preds

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        in_flight = collections.deque()
        pred = torch.tensor([[0.5], [-1.0], [2.0]], dtype=dtype)
        inference._queue_result(in_flight, (0, True, pred, np.arange(3), np.zeros((3, 2), np.int64)), None)
        pending = in_flight.popleft()
        writer = Writer()
        inference._collect_result(pending, {}, writer, "out", 0)
        assert writer.preds.dtype == np.float32 and writer.preds.tolist() == [0.5, -1.0, 2.0]
        assert not np.shares_memory(writer.preds, pending.host_pred.numpy())


def test_the_loops_own_error_is_raised_when_the_writer_also_fails(save_dict, fake_cuda, tmp_path, monkeypatch, caplog):  # noqa: F811
    model = _engine(save_dict, "eager")

    class FailingModel:
        block_len = model.block_len
        calls = 0

        def __call__(self, *args, **kwargs):
            FailingModel.calls += 1
            if FailingModel.calls == 4:
                raise ValueError("model failed")
            return model(*args, **kwargs)

    def failing_write(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(inference.np, "savez_compressed", failing_write)
    loader = _Loader([_shard(16, i) for i in range(6)])
    with pytest.raises(ValueError, match="model failed"):
        inference._inference_loop_pipelined(_args(tmp_path), 0, torch.device("cpu"), FailingModel(), loader)
    assert "also failed" in caplog.text and "No space left" in caplog.text


def test_a_failed_write_is_not_logged_as_files_still_to_be_written(fake_cuda, tmp_path, monkeypatch, caplog):
    """When the loop stops because a write failed, the writer writes nothing more, and the log says so."""
    writers = []

    class Writer(inference._ShardWriter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            writers.append(self)

    class Model:
        block_len = 9
        calls = 0

        def __call__(self, kmer, sig, seg_len, dwell_bq):
            Model.calls += 1
            if Model.calls == 5:  # shard 0 went to the writer after call 3: wait until its write has failed
                assert writers[0]._done.wait(10)
            return kmer[:, :1].float()  # [B, 1]

    def failing_write(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(inference, "_ShardWriter", Writer)
    monkeypatch.setattr(inference.np, "savez_compressed", failing_write)
    loader = _Loader([_shard(16, i) for i in range(8)])
    with pytest.raises(RuntimeError, match="Writing an output file failed"):
        inference._inference_loop_pipelined(_args(tmp_path), 0, torch.device("cpu"), Model(), loader)
    assert "Rank 0 stopped on RuntimeError(\"Writing an output file failed" in caplog.text, caplog.text
    assert "the output writer has stopped, so no more output files are written" in caplog.text
    assert "already queued are written first" not in caplog.text
