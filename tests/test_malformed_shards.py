"""An input shard that can be read but is malformed (an array missing, of the wrong dtype or shape, or a row longer
than its signal) is skipped with a warning, as an unreadable shard is: the run goes on, no output file is written for
it, --resume treats it the same way, and the other shards' outputs do not change."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_pipelined_loop import fake_cuda  # noqa: E402, F401  (a fixture)
from test_v12_model import WEIGHTS, _engine, _run_args, _write_prep, save_dict  # noqa: E402, F401

from deeprm.inference import inference, inference_dataloader  # noqa: E402

N_ROWS = 12


def _too_long_row(seg):
    seg = seg.copy()
    seg[3] = 12  # 17 * 12 = 204 windows in a row whose signal_token has 200
    return seg


def _negative_length(seg):
    seg = seg.astype(np.int32)
    seg[0, 0] = -1
    return seg


# name: (change to shard 1, words of the warning). A change maps an array to its new value; None removes the array.
MALFORMED = {
    "segment width": ({"segment_len_arr": lambda a: a[:, :16]}, "segment_len_arr has 16 columns; this model needs 17"),
    "dwell width": ({"dwell_pore_token": lambda a: a[:, :16]}, "dwell_pore_token has 16 columns"),
    "kmer width": ({"kmer_token": lambda a: a[:, :20]}, "kmer_token has 20 columns; this model needs 21"),
    "narrow signal": ({"signal_token": lambda a: a[:, :24]}, "signal_token has 24 samples per row"),
    "wide signal": (
        {"signal_token": lambda a: np.zeros((len(a), 2048 * 6 + 30), np.float32)},
        "2049 signal windows per row; this model takes at most 2048",
    ),
    "fewer labels": ({"label_id": lambda a: a[:-2]}, f"read_id has {N_ROWS} rows, but label_id has {N_ROWS - 2}"),
    "fewer signal rows": (
        {"signal_token": lambda a: a[:-1]},
        f"signal_token has {N_ROWS - 1} rows, but label_id has {N_ROWS}",
    ),
    "1-D read_id": ({"read_id": lambda a: a[:, 0]}, "read_id has shape (12,); it must have 2 dimensions"),
    "text kmer": ({"kmer_token": lambda a: np.full(a.shape, "A")}, "kmer_token has dtype <U1"),
    "missing array": ({"segment_len_arr": None}, "lacks the array(s) segment_len_arr"),
    "too long row": (
        {"segment_len_arr": _too_long_row},
        "in 1 row(s), segment_len_arr adds up to more than the 200 signal windows",
    ),
    "negative length": ({"segment_len_arr": _negative_length}, "segment_len_arr holds a negative length"),
}


def _break_shard(path, changes):
    with np.load(path) as z:
        arrays = {key: z[key] for key in z.files}
    for key, change in changes.items():
        if change is None:
            del arrays[key]
        else:
            arrays[key] = change(arrays[key])
    np.savez(path, **arrays)


def _prep_folder(folder, broken=None):
    """Three shards as deeprm call prep names them; shard 1 is changed by `broken`."""
    folder.mkdir()
    for i in range(3):
        _write_prep(folder / f"{i}.npz", n=N_ROWS, seed=i)
    if broken is not None:
        _break_shard(folder / "1.npz", broken)
    return folder


def _call_run(data, out, resume=False):
    out.mkdir(exist_ok=True)
    args = _run_args(data, out, WEIGHTS)
    args.resume = resume
    inference.write_run_info(args)
    inference.inference_worker(0, vars(args))
    return _files(out)


def _files(folder):
    """The per-read files of an output folder, by name."""
    found = {}
    for name in sorted(os.listdir(folder)):
        assert not name.endswith(".part"), name
        if name.startswith("inference_"):
            with np.load(os.path.join(folder, name)) as f:
                found[name] = {key: f[key] for key in ("label_id", "read_id", "pred")}
    return found


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    """The outputs of a run on three valid shards."""
    root = tmp_path_factory.mktemp("clean")
    return _call_run(_prep_folder(root / "prep"), root / "out")


def _same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        assert a[key].dtype == b[key].dtype and np.array_equal(a[key].view(np.uint8), b[key].view(np.uint8)), key


@pytest.mark.parametrize("case", list(MALFORMED))
def test_malformed_shard_is_skipped_and_the_run_completes(clean, tmp_path, caplog, monkeypatch, case):
    changes, words = MALFORMED[case]
    monkeypatch.setattr(inference_dataloader, "READ_RETRIES", 1)  # an unreadable shard would cost 1 s, not 6 s
    data = _prep_folder(tmp_path / "prep", changes)
    with caplog.at_level("WARNING"):
        found = _call_run(data, tmp_path / "out")
    # no file for shard 1; the other shards keep their numbers and their outputs, bit for bit
    assert sorted(found) == ["inference_0_0.npz", "inference_0_2.npz"]
    for name in found:
        _same(found[name], clean[name])
    assert f"Skipping input shard {data / '1.npz'}: " in caplog.text and words in caplog.text
    assert "--resume" in caplog.text
    assert "Failed to read" not in caplog.text, "a malformed shard is not read again"
    assert "skipped 1 input shard(s)" in caplog.text


@pytest.mark.parametrize("case", ["dwell width", "too long row", "fewer signal rows"])
def test_both_loops_skip_a_malformed_shard(save_dict, fake_cuda, tmp_path, case):  # noqa: F811
    model = _engine(save_dict, "fast")
    data = _prep_folder(tmp_path / "prep", MALFORMED[case][0])
    found = {}
    for name, loop in (("sync", inference._inference_loop_sync), ("pipelined", inference._inference_loop_pipelined)):
        out = tmp_path / name
        out.mkdir()
        loader = inference_dataloader.load_dataset(str(data), 0, 1, worker=0, pin_memory=False)
        args = _run_args(data, out, WEIGHTS)
        args = {**vars(args), "output_logits": True, "dwell_bq_channels": None}
        loop(args, 0, torch.device("cpu"), model, loader)
        found[name] = _files(out)
    assert sorted(found["sync"]) == sorted(found["pipelined"]) == ["inference_0_0.npz", "inference_0_2.npz"]
    for name in found["sync"]:
        _same(found["pipelined"][name], found["sync"][name])


def test_resume_runs_again_from_a_malformed_shard_once_it_is_re_created(clean, tmp_path, caplog):
    data = _prep_folder(tmp_path / "prep", MALFORMED["too long row"][0])
    out = tmp_path / "out"
    assert sorted(_call_run(data, out)) == ["inference_0_0.npz", "inference_0_2.npz"]
    _write_prep(data / "1.npz", n=N_ROWS, seed=1)  # the shard re-created under the same name
    with caplog.at_level("WARNING"):
        found = _call_run(data, out, resume=True)
    # as after an unreadable shard: the run starts again from the skipped shard, and the shards after it run again
    assert "Removed shard indices >= 1: [2]" in caplog.text
    assert sorted(found) == sorted(clean)
    for name in found:
        _same(found[name], clean[name])


def test_a_folder_of_malformed_shards_stops_the_run(tmp_path, caplog):
    """Shards that are malformed in different ways: each is skipped with its warning, then the run stops."""
    data = tmp_path / "prep"
    data.mkdir()
    for i, case in enumerate(["segment width", "kmer width", "dwell width"]):
        _write_prep(data / f"{i}.npz", n=N_ROWS, seed=i)
        _break_shard(data / f"{i}.npz", MALFORMED[case][0])
    with caplog.at_level("WARNING"), pytest.raises(RuntimeError, match="could use none of its 3 input shard"):
        _call_run(data, tmp_path / "out")
    assert caplog.text.count("Skipping input shard") == 3


@pytest.mark.parametrize("n_shards", [1, 5])
@pytest.mark.parametrize("case", ["segment width", "wide signal"])
def test_a_folder_that_does_not_fit_the_model_stops_at_the_start(tmp_path, caplog, case, n_shards):
    """A folder that prep wrote with other geometry options (--cb-len / --kmer-len, or a --max-token-len over 2048)
    fits in none of its shards: one error before any shard is read, not one warning per shard."""
    changes, words = MALFORMED[case]
    data = tmp_path / "prep"
    data.mkdir()
    for i in range(n_shards):
        _write_prep(data / f"{i}.npz", n=N_ROWS, seed=i)
        _break_shard(data / f"{i}.npz", changes)
    with caplog.at_level("WARNING"), pytest.raises(ValueError, match="do not fit this model") as refused:
        _call_run(data, tmp_path / "out")
    assert words in str(refused.value)
    assert "--cb-len 21, --kmer-len 5 and a --max-token-len of at most 2048" in str(refused.value)
    assert "Skipping input shard" not in caplog.text, "no shard was read"


def test_the_header_check_takes_valid_shards_and_leaves_unreadable_ones_to_the_reader(tmp_path):
    _write_prep(tmp_path / "plain.npz", n=N_ROWS)
    with np.load(tmp_path / "plain.npz") as z:
        arrays = {key: z[key] for key in z.files}
    np.savez_compressed(tmp_path / "compressed.npz", **arrays)  # as prep writes them
    np.savez(tmp_path / "empty.npz", **{key: value[:0] for key, value in arrays.items()})
    (tmp_path / "broken.npz").write_bytes(b"not an npz file")
    for name in ("plain", "compressed", "empty", "broken"):
        assert inference_dataloader.folder_problem([str(tmp_path / f"{name}.npz")], max_seq_len=2048) is None, name
    # the first, middle and last shard must all fail in the same way
    _write_prep(tmp_path / "narrow.npz", n=N_ROWS)
    _break_shard(tmp_path / "narrow.npz", MALFORMED["segment width"][0])
    narrow, plain = str(tmp_path / "narrow.npz"), str(tmp_path / "plain.npz")
    assert inference_dataloader.folder_problem([narrow, plain, narrow]) is None
    assert inference_dataloader.folder_problem([narrow, plain, narrow, narrow]) == (
        [narrow, narrow, narrow],
        MALFORMED["segment width"][1],
    )


def test_valid_shards_pass_the_check():
    rng = np.random.default_rng(0)
    seg = rng.integers(0, 12, size=(5, 17)).astype(np.uint16)
    seg[0] = 0  # a row with no signal
    seg[1, 8] = 200 - (int(seg[1].sum()) - int(seg[1, 8]))  # a row exactly as long as its signal
    shard = {
        "read_id": np.zeros((5, 2), np.int64),
        "label_id": np.arange(5, dtype=np.int64),
        "segment_len": seg,
        "signal_token": np.zeros((5, 1224), np.float32),
        "kmer_token": np.full((5, 21), ord("A"), np.uint8),
        "dwell_motor_token": np.zeros((5, 17), np.float32),
        "dwell_pore_token": np.zeros((5, 17), np.float32),
        "bq_token": np.zeros((5, 17), np.uint8),
    }
    assert inference_dataloader.shard_problem(shard, max_seq_len=2048) is None
    # an empty shard is not malformed: the loop skips it with its own warning
    empty = {key: value[:0] for key, value in shard.items()}
    assert inference_dataloader.shard_problem(empty, max_seq_len=2048) is None
    # a prep run with another --max-token-len (here 120) writes a narrower signal_token, which the model takes
    narrow = {**shard, "segment_len": np.minimum(seg, 7), "signal_token": np.zeros((5, 124 * 6), np.float32)}
    assert inference_dataloader.shard_problem(narrow, max_seq_len=2048) is None
