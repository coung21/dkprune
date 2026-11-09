import sys
from pathlib import Path

import pytest
import yaml

import run as cli
from dkprune import config, darknet, graph, loop, summary
from dkprune.io import cfg as cfgio
from dkprune.io import weights as wt
from tests.test_prune import random_params

CFGS = Path(__file__).parent / "cfgs"
FAKE = Path(__file__).parent / "fake_darknet.py"


# ---------------------------------------------------------------- darknet.py

def test_parse_map_strips_colour_and_takes_last():
    text = ("mean average precision (mAP@0.50)=10.00%\n"
            "-> mean average precision (mAP@0.50)=\x1b[1;34m73.45%\x1b[0m\n")
    assert darknet.parse_map(text) == pytest.approx(0.7345)


def test_parse_map_missing():
    with pytest.raises(darknet.DarknetError, match="Could not find mAP"):
        darknet.parse_map("Training complete.")


def test_write_data_file(tmp_path):
    src = tmp_path / "a.data"
    src.write_text("classes = 5\ntrain = /x/train.txt\nvalid = /x/valid.txt\n# backup = old\nbackup = /old\n")
    dst = tmp_path / "b.data"
    darknet.write_data_file(src, dst, tmp_path / "bk", valid=Path("/x/test.txt"))
    d = darknet.read_data_file(dst)
    assert d == {"classes": "5", "train": "/x/train.txt", "valid": "/x/test.txt", "backup": str(tmp_path / "bk")}
    assert "# backup = old" in dst.read_text()                 # comment giữ nguyên


def test_write_data_file_adds_missing_backup(tmp_path):
    src = tmp_path / "a.data"
    src.write_text("classes = 5\n")
    darknet.write_data_file(src, tmp_path / "b.data", tmp_path / "bk")
    assert darknet.read_data_file(tmp_path / "b.data")["backup"] == str(tmp_path / "bk")


def test_find_weights_preference(tmp_path):
    for s in ("_last", "_final"):
        (tmp_path / f"r1{s}.weights").write_bytes(b"x")
    assert darknet.find_weights(tmp_path, "r1").name == "r1_final.weights"
    (tmp_path / "r1_best.weights").write_bytes(b"x")
    assert darknet.find_weights(tmp_path, "r1").name == "r1_best.weights"
    with pytest.raises(darknet.DarknetError, match="valid="):
        darknet.find_weights(tmp_path, "r2")


def test_darknet_nonzero_exit(tmp_path):
    dk = darknet.Darknet((sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"), tmp_path)
    with pytest.raises(darknet.DarknetError, match="exit with code 3"):
        dk.map(tmp_path / "a.data", tmp_path / "a.cfg", tmp_path / "a.weights", tmp_path / "log.txt")
    assert "boom" in (tmp_path / "log.txt").read_text()


def test_darknet_missing_program(tmp_path):
    dk = darknet.Darknet.from_string("khong-co-chuong-trinh-nay", tmp_path)
    with pytest.raises(darknet.DarknetError, match="darknet_cmd"):
        dk.map(tmp_path / "a", tmp_path / "b", tmp_path / "c", tmp_path / "log.txt")


# ---------------------------------------------------------------- [net] cho fine-tune

def test_net_edits():
    c = cfgio.set_values(cfgio.load(CFGS / "yolov7-tiny-gc10.cfg"), loop.net_edits(2000, 0.001, 200))
    net = {o.key: o.value for o in c.net.options}
    assert net["max_batches"] == "2000" and net["steps"] == "1600,1800"
    assert net["learning_rate"] == "0.001" and net["burn_in"] == "200"
    assert net["policy"] == "steps" and net["scales"] == ".1,.1"


# ---------------------------------------------------------------- config

def make_project(tmp_path: Path, **overrides) -> Path:
    """Tạo dữ liệu giả: cfg GC10, weights ngẫu nhiên, file .data, và config.yaml."""
    c = cfgio.load(CFGS / "yolov7-tiny-gc10.cfg")
    g = graph.build(c)
    (tmp_path / "data").mkdir()
    cfgio.save(c, tmp_path / "data" / "model.cfg")
    w = graph.write_params(g, random_params(g), wt.Header(0, 2, 5, 384000))
    wt.save(w, tmp_path / "data" / "model.weights")
    (tmp_path / "data" / "my.data").write_text(
        "classes = 10\ntrain = train.txt\nvalid = valid.txt\nnames = my.names\nbackup = backup/\n")
    conf = {
        "data": "data/my.data", "cfg": "data/model.cfg", "weights": "data/model.weights",
        "run_dir": "runs/r1", "darknet_cmd": f"{sys.executable} {FAKE}",
        "rounds": 3, "max_map_drop": 0.02,
        "prune": {"ratio": 0.16, "skip": [0]},
        "finetune": {"max_batches": 1000, "last_max_batches": 3000, "learning_rate": 0.001, "burn_in": 100},
    }
    conf.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(conf), encoding="utf-8")
    return path


def test_config_resolves_relative_paths(tmp_path):
    rc = config.load(make_project(tmp_path))
    assert rc.data == (tmp_path / "data" / "my.data").resolve()
    assert rc.darknet_workdir == rc.data.parent
    assert rc.prune.skip == (0,) and rc.finetune.last_max_batches == 3000


def test_config_errors(tmp_path):
    p = make_project(tmp_path)
    raw = yaml.safe_load(p.read_text())
    del raw["cfg"]
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(config.ConfigError, match="required key 'cfg' is missing"):
        config.load(p)
    raw["cfg"] = "data/khong-co.cfg"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(config.ConfigError, match="cannot find file cfg"):
        config.load(p)


# ---------------------------------------------------------------- vòng lặp với Darknet giả

@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    state = tmp_path / "fake_state"
    monkeypatch.setenv("FAKE_STATE", str(state))

    def counts():
        if not state.exists():
            return {}
        return {k: int(v) for k, v in (l.split("=") for l in state.read_text().splitlines())}
    return counts


def test_loop_stops_at_threshold(tmp_path, monkeypatch, fake_env):
    # mAP: gốc 80%, vòng 1: 79.5%, vòng 2: 78.5% (sụt 1.5 điểm), vòng 3: 70% (sụt 10 điểm -> dừng)
    monkeypatch.setenv("FAKE_MAPS", "0.80,0.795,0.785,0.70")
    rc = config.load(make_project(tmp_path))
    logs: list[str] = []
    res = loop.run(rc, log=logs.append)

    assert [r.round for r in res] == [0, 1, 2, 3]
    assert res[0].params > res[1].params > res[2].params > res[3].params
    assert summary.choose(res, rc.max_map_drop).round == 2
    assert fake_env() == {"map": 4, "train": 3}
    run = rc.run_dir
    for k in (1, 2, 3):
        assert (run / f"round_{k}_best.weights").exists()
        assert (run / "logs" / f"round_{k}_train.log").exists()
        assert (run / "logs" / f"charts_{k}" / "chart.png").exists()   # biểu đồ được chuyển vào logs
    # vòng cuối (3) train lâu hơn
    for k, expect in ((1, "1000"), (3, "3000")):
        mb = cfgio.load(run / f"round_{k}.cfg").net.find("max_batches")
        assert mb is not None and mb.value == expect
    text = (run / "summary.md").read_text()
    assert "**choose**" in text and "gone over" in text
    assert any("stopping" in m for m in logs)


def test_loop_resume_does_not_redo_rounds(tmp_path, monkeypatch, fake_env):
    monkeypatch.setenv("FAKE_MAPS", "0.80,0.79,0.785,0.78")
    path = make_project(tmp_path)
    first = loop.run(config.load(path), log=lambda m: None)
    assert fake_env()["train"] == 3
    again = loop.run(config.load(path), log=lambda m: None)
    assert again == first
    assert fake_env()["train"] == 3                              # không train lại


def test_loop_resumes_after_crash(tmp_path, monkeypatch, fake_env):
    monkeypatch.setenv("FAKE_MAPS", "0.80,0.79,0.785,0.78")
    monkeypatch.setenv("FAKE_FAIL_TRAIN", "2")                   # lần train thứ 2 (vòng 2) bị lỗi
    path = make_project(tmp_path)
    with pytest.raises(darknet.DarknetError, match="out of memory"):
        loop.run(config.load(path), log=lambda m: None)
    rc = config.load(path)
    assert (rc.run_dir / "round_1.json").exists() and not (rc.run_dir / "round_2.json").exists()

    monkeypatch.delenv("FAKE_FAIL_TRAIN")
    res = loop.run(rc, log=lambda m: None)                        # chạy tiếp từ vòng 2
    assert [r.round for r in res] == [0, 1, 2, 3]
    assert fake_env()["train"] == 4                              # vòng 1 không bị train lại


def test_loop_requires_valid_in_data(tmp_path, monkeypatch, fake_env):
    path = make_project(tmp_path)
    data = tmp_path / "data" / "my.data"
    data.write_text(data.read_text().replace("valid = valid.txt\n", ""))
    with pytest.raises(ValueError, match="valid="):
        loop.run(config.load(path), log=lambda m: None)


# ---------------------------------------------------------------- dòng lệnh

def test_cli_inspect_and_verify(tmp_path, capsys):
    make_project(tmp_path)
    d = tmp_path / "data"
    assert cli.main(["inspect", str(d / "model.cfg"), str(d / "model.weights")]) == 0
    out = capsys.readouterr().out
    assert "55/58 convs prunable" in out and "cfg and weights match" in out
    assert cli.main(["verify", str(d / "model.cfg"), str(CFGS / "yolov7-tiny-gc10.cfg")]) == 2   # file sai


def test_cli_prune(tmp_path, monkeypatch, fake_env, capsys):
    monkeypatch.setenv("FAKE_MAPS", "0.80,0.79,0.785,0.78")
    path = make_project(tmp_path)
    assert cli.main(["prune", str(path)]) == 0
    assert (tmp_path / "runs" / "r1" / "summary.md").exists()
    assert (tmp_path / "runs" / "r1" / "config.yaml").exists()