"""A row with a NaN or infinite input value: its prediction is NaN and the other rows of its batch are not affected.

CPU only. fa4 runs with the CPU stand-in for FlashAttention-4 (fa4_cpu_standin), which reads keys and values in whole
128-row tiles as FlashAttention-4 does, so a NaN that reaches the neighbouring rows' tiles shows here as on a GPU.
"""

import contextlib
import logging
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_pipelined_loop import _args, _files, fake_cuda  # noqa: E402, F401  (fake_cuda is a fixture)
from test_v12_model import _engine, _synthetic_inputs, save_dict  # noqa: E402, F401  (save_dict is a fixture)

from deeprm.inference import inference, inference_dataloader  # noqa: E402

N = 40  # rows per shard; --batch 16 makes model calls of 16, 16 and 8 rows
FLOATS = ("signal_token", "dwell_motor_token", "dwell_pore_token")
# row -> (array, value): a NaN and a +Inf signal value in the middle of a model call, and a -Inf dwell value
BAD = {5: ("signal_token", np.nan), 20: ("signal_token", np.inf), 33: ("dwell_pore_token", -np.inf)}


def _arrays(n, seed):
    """One shard's arrays, as deeprm call prep writes them."""
    rng = np.random.default_rng(seed)
    kmer, signal, seg = _synthetic_inputs(n, seed)
    return {
        "segment_len_arr": seg.numpy().astype(np.uint16),
        "signal_token": signal.numpy(),
        "kmer_token": kmer.numpy().astype(np.uint8),
        "dwell_motor_token": rng.random((n, 17)).astype(np.float32),
        "dwell_pore_token": rng.random((n, 17)).astype(np.float32),
        "bq_token": rng.integers(0, 40, size=(n, 17)).astype(np.uint8),
        "label_id": np.arange(n, dtype=np.int64) + 1 + 1000 * seed,
        "read_id": rng.integers(0, 2**62, size=(n, 2), dtype=np.int64),
    }


def _prep_dir(folder, zeroed=False):
    """0.npz: a clean shard. 1.npz: the BAD values, or, with zeroed=True, all float values of the BAD rows set to 0."""
    folder.mkdir()
    np.savez(folder / "0.npz", **_arrays(N, 1))
    shard = _arrays(N, 2)
    for row, (key, value) in BAD.items():
        if zeroed:
            for float_key in FLOATS:
                shard[float_key][row] = 0
        else:
            shard[key][row, 7] = value
    np.savez(folder / "1.npz", **shard)
    return folder


def _run(loop, model, data, out, batch=16, worker=0, device="cpu"):
    """Runs one loop over the shards in `data` through the real data loader; returns the files it wrote."""
    out.mkdir()
    loader = inference_dataloader.load_dataset(str(data), 0, 1, worker=worker, pin_memory=False)
    run = inference._inference_loop_sync if loop == "sync" else inference._inference_loop_pipelined
    run(_args(out, batch), 0, torch.device(device), model, loader)
    return _files(out)


def _bad_mask():
    bad = np.zeros(N, dtype=bool)
    bad[list(BAD)] = True
    return bad


def _assert_isolated(got, ref):
    """The BAD rows are NaN; every other row is bit-identical to the run where the BAD rows were zeros."""
    assert sorted(got) == sorted(ref) == ["inference_0_0.npz", "inference_0_1.npz"]
    bad = _bad_mask()
    pred, ref_pred = got["inference_0_1.npz"]["pred"], ref["inference_0_1.npz"]["pred"]
    assert np.isfinite(ref_pred).all()  # rows of zeros give finite predictions, so a NaN that spread would show
    changed = np.flatnonzero(~bad & (pred.view(np.uint32) != ref_pred.view(np.uint32)))
    assert changed.size == 0, f"other rows changed: {changed.tolist()}, NaN: {np.flatnonzero(np.isnan(pred)).tolist()}"
    assert np.isnan(pred[bad]).all(), pred[bad]
    clean, ref_clean = got["inference_0_0.npz"]["pred"], ref["inference_0_0.npz"]["pred"]
    assert np.array_equal(clean.view(np.uint32), ref_clean.view(np.uint32))
    for name in got:
        for key in ("label_id", "read_id"):
            assert np.array_equal(got[name][key], ref[name][key])


@pytest.mark.parametrize("loop", ["sync", "pipelined"])
@pytest.mark.parametrize("engine", ["fa4", "fast", "eager"])
def test_a_row_with_a_nonfinite_value_is_nan_and_the_others_are_unchanged(
    save_dict, fake_cuda, tmp_path, caplog, engine, loop  # noqa: F811
):
    import fa4_cpu_standin

    fa4_cpu_standin.enable()
    model = _engine(save_dict, engine)
    with caplog.at_level(logging.WARNING):
        got = _run(loop, model, _prep_dir(tmp_path / "bad"), tmp_path / "out")
    ref = _run(loop, model, _prep_dir(tmp_path / "zeroed", zeroed=True), tmp_path / "ref")
    _assert_isolated(got, ref)
    # one warning, for the shard that has such rows, with its file and the row count
    found = [r.getMessage() for r in caplog.records if "NaN or infinite" in r.getMessage()]
    assert len(found) == 1 and "1.npz" in found[0] and f"{len(BAD)} of {N} rows" in found[0], found


@pytest.mark.parametrize("loop", ["sync", "pipelined"])
def test_the_rows_stay_isolated_after_out_of_memory_and_through_a_worker_process(
    save_dict, fake_cuda, tmp_path, monkeypatch, loop  # noqa: F811
):
    """A GPU that holds at most 8 rows: the 16-row batches are run again in parts of 8 rows (the out-of-memory path),
    and the shards come from a DataLoader worker process. The reference is a run of 8-row batches on zeroed rows."""
    import fa4_cpu_standin

    fa4_cpu_standin.enable()
    model = _engine(save_dict, "fa4")

    class SmallGpu:
        block_len = model.block_len
        ACCEPTS_COUNTS = model.ACCEPTS_COUNTS

        def __call__(self, kmer, *args, **kwargs):
            if kmer.shape[0] > 8:
                raise torch.OutOfMemoryError("CUDA out of memory (test)")
            return model(kmer, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    device = "cpu"
    if loop == "sync":
        # The sync loop retries an out-of-memory error on a GPU only: name a GPU, keep the tensors on the CPU.
        device = "cuda"
        to_device = inference.to_device
        monkeypatch.setattr(inference, "to_device", lambda data, dev, ch: to_device(data, torch.device("cpu"), ch))
        monkeypatch.setattr(inference, "autocast", lambda **kwargs: contextlib.nullcontext())
    got = _run(loop, SmallGpu(), _prep_dir(tmp_path / "bad"), tmp_path / "out", worker=1, device=device)
    ref = _run(loop, model, _prep_dir(tmp_path / "zeroed", zeroed=True), tmp_path / "ref", batch=8)
    _assert_isolated(got, ref)


def test_a_clean_shard_passes_unchanged_and_gives_the_same_output(save_dict, tmp_path, monkeypatch):  # noqa: F811
    arrays = _arrays(N, 3)
    np.savez(tmp_path / "0.npz", **arrays)
    reader = inference_dataloader.NanoporeDatasetIterator([str(tmp_path / "0.npz")])
    data = next(reader)
    reader.close()
    assert inference_dataloader.NONFINITE_KEY not in data
    renamed = {"segment_len": "segment_len_arr"}
    for key, values in data.items():
        expected = arrays[renamed.get(key, key)]
        assert values.dtype == expected.dtype and np.array_equal(values.view(np.uint8), expected.view(np.uint8))
    assert inference_dataloader.NONFINITE_KEY not in inference_dataloader.collate_fn(data)

    # The same files as a run without the check (the code before it).
    model = _engine(save_dict, "fast")
    data_dir = tmp_path / "prep"
    data_dir.mkdir()
    np.savez(data_dir / "0.npz", **arrays)
    with_check = _run("sync", model, data_dir, tmp_path / "with")
    monkeypatch.setattr(inference_dataloader, "zero_nonfinite_rows", lambda data, path: 0)
    without = _run("sync", model, data_dir, tmp_path / "without")
    for key in ("label_id", "read_id", "pred"):
        a, b = with_check["inference_0_0.npz"][key], without["inference_0_0.npz"][key]
        assert a.dtype == b.dtype and np.array_equal(a.view(np.uint8), b.view(np.uint8))


def test_zero_nonfinite_rows_leaves_malformed_and_empty_shards_to_the_model():
    def shard(n, signal):
        return {
            "label_id": np.arange(n),
            "signal_token": signal,
            "dwell_motor_token": np.zeros((n, 17), np.float32),
            "dwell_pore_token": np.zeros((n, 17), np.float32),
        }

    empty = shard(0, np.zeros((0, 1224), np.float32))
    assert inference_dataloader.zero_nonfinite_rows(empty, "empty.npz") == 0
    assert inference_dataloader.NONFINITE_KEY not in empty
    # an array of another row count or rank, or of integers, is not checked and not changed (the model refuses it)
    for signal in (np.full((3, 1224), np.nan, np.float32), np.full(4, np.nan, np.float32), np.ones((4, 9), np.int16)):
        data = shard(4, signal)
        assert inference_dataloader.zero_nonfinite_rows(data, "odd.npz") == 0
        assert data["signal_token"] is signal and inference_dataloader.NONFINITE_KEY not in data
    # float64 and a 3-D array are checked as well
    data = shard(4, np.ones((4, 2, 3), np.float64))
    data["signal_token"][2, 1, 0] = np.inf
    assert inference_dataloader.zero_nonfinite_rows(data, "deep.npz") == 1
    assert data[inference_dataloader.NONFINITE_KEY].tolist() == [False, False, True, False]
    assert not data["signal_token"][2].any() and data["signal_token"][[0, 1, 3]].all()


def test_a_float64_value_beyond_the_float32_maximum_counts_as_infinite():
    """collate_fn gives the model float32: a float64 value over about 3.4e38 would become Inf there, so it is marked."""
    signal = np.ones((3, 1224), np.float64)
    signal[1, 5] = 1e39  # Inf as float32
    signal[2, 5] = -3e38  # still finite as float32
    data = {
        "label_id": np.arange(3),
        "signal_token": signal,
        "dwell_motor_token": np.zeros((3, 17), np.float32),
        "dwell_pore_token": np.zeros((3, 17), np.float32),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the float32 cast must not warn about the overflow it looks for
        assert inference_dataloader.zero_nonfinite_rows(data, "big.npz") == 1
    assert data[inference_dataloader.NONFINITE_KEY].tolist() == [False, True, False]
    assert not data["signal_token"][1].any() and data["signal_token"][2, 5] == -3e38
    batch = {**_arrays(3, 0), **data}
    batch["segment_len"] = batch.pop("segment_len_arr")
    assert torch.isfinite(inference_dataloader.collate_fn(batch)["signal_token"]).all()  # the model's float32 input
