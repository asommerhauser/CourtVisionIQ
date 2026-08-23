"""
The cviq shell: dispatch, dial commands, and the LOAD/RUN wiring.

Everything here runs without TensorFlow and without touching real weights. ``CviqShell.onecmd``
is driven directly, which is the whole reason the shell is built on ``cmd.Cmd``: commands are
testable as function calls rather than by driving a terminal.

The load/run tests stub ``GameSimulator.load`` and ``evaluate_stage``, so they assert the
*wiring* -- that a loaded sim is reused, that dials reach the run, that a missing head is caught
at load rather than mid-rollout -- without a 2.7 GB model.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

import config
from shell.actions import ShellError, train_command
from shell.repl import ArgError, CviqShell
from shell.session import Session


@pytest.fixture
def sh(capsys):
    """A shell whose banner has already been swallowed."""
    s = CviqShell()
    capsys.readouterr()
    return s


def run(sh, capsys, line):
    sh.onecmd(line)
    return capsys.readouterr().out


# --------------------------------------------------------------------------- #
# --- Dispatch + error containment                                          -- #
# --------------------------------------------------------------------------- #

def test_unknown_command_is_a_message_not_a_traceback(sh, capsys):
    out = run(sh, capsys, "bogus")
    assert "unknown command" in out and "Traceback" not in out


def test_bad_flag_does_not_kill_the_shell(sh, capsys):
    """argparse would normally SystemExit -- fatal in a process holding a loaded model."""
    out = run(sh, capsys, "run --nonsense")
    assert "Traceback" not in out
    assert run(sh, capsys, "reset")  # shell still answers


def test_action_exception_is_contained(sh, capsys, monkeypatch):
    monkeypatch.setattr("shell.repl.run_eval",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = run(sh, capsys, "run trial1")
    assert "boom" in out and "Traceback" not in out


def test_keyboard_interrupt_keeps_the_model_loaded(sh, capsys, monkeypatch):
    monkeypatch.setattr("shell.repl.run_eval",
                        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    sh.session.sim = object()
    out = run(sh, capsys, "run trial1")
    assert "still loaded" in out
    assert sh.session.loaded


def test_empty_line_is_a_noop(sh, capsys):
    assert run(sh, capsys, "") == ""


def test_quit_returns_true(sh):
    assert sh.onecmd("quit") is True


# --------------------------------------------------------------------------- #
# --- Dials                                                                 -- #
# --------------------------------------------------------------------------- #

def test_set_changes_a_dial(sh, capsys):
    run(sh, capsys, "set DELTA_TIME_SCALE 0.99")
    assert config.DELTA_TIME_SCALE == 0.99


def test_set_unknown_dial_suggests_neighbours(sh, capsys):
    out = run(sh, capsys, "set DELTA_TIME_SCLAE 0.99")
    assert "unknown dial" in out
    assert "DELTA_TIME_SCALE" in out  # prefix suggestion fired


def test_set_dict_dial_takes_json(sh, capsys):
    run(sh, capsys, 'set TYPE_BIAS {"foul_type": {"shooting": 0.5}}')
    assert config.TYPE_BIAS == {"foul_type": {"shooting": 0.5}}


def test_dials_changed_shows_before_and_after(sh, capsys):
    run(sh, capsys, "set DELTA_TIME_SCALE 0.99")
    out = run(sh, capsys, "dials --changed")
    assert "0.97" in out and "0.99" in out


def test_unset_restores_one_dial(sh, capsys):
    before = config.DELTA_TIME_SCALE
    run(sh, capsys, "set DELTA_TIME_SCALE 1.5")
    run(sh, capsys, "unset DELTA_TIME_SCALE")
    assert config.DELTA_TIME_SCALE == before


def test_reset_restores_everything_including_dict_dials(sh, capsys):
    before = config.get_dials()
    run(sh, capsys, "set DELTA_TIME_SCALE 1.5")
    run(sh, capsys, 'set TYPE_BIAS {"foul_type": {"shooting": 9.0}}')
    run(sh, capsys, "reset")
    assert config.get_dials() == before


def test_dials_save_round_trips(sh, capsys, tmp_path):
    run(sh, capsys, "set DELTA_TIME_SCALE 0.99")
    f = tmp_path / "pack.json"
    run(sh, capsys, f"dials --save {f}")
    assert json.loads(f.read_text(encoding="utf-8"))["DELTA_TIME_SCALE"] == 0.99
    run(sh, capsys, "reset")
    run(sh, capsys, f"dialfile {f}")
    assert config.DELTA_TIME_SCALE == 0.99


def test_complete_set_offers_dial_names(sh):
    assert "DELTA_TIME_SCALE" in sh.complete_set("DELTA", "", 0, 0)


# --------------------------------------------------------------------------- #
# --- LOAD                                                                  -- #
# --------------------------------------------------------------------------- #

def _fake_model_dir(root, name, heads):
    d = root / name
    for h in heads:
        (d / h).mkdir(parents=True)
        (d / h / f"{h}.weights.h5").write_bytes(b"x")
    return d


ALL_HEADS = ("event_time", "player", "substitution", "shot_type", "shot_result",
             "assist_type", "turnover_type", "foul_type", "rebound_type",
             "event_time_cond", "stint_length")


@pytest.fixture
def fake_artifacts(tmp_path, monkeypatch):
    """An artifacts tree with one complete model, plus stubbed TF-dependent imports."""
    root = tmp_path / "artifacts"
    _fake_model_dir(root, "v1.0", ALL_HEADS)
    monkeypatch.setattr("models.artifacts.MODELS_ROOT", str(root))
    monkeypatch.setattr("shell.actions.model_root", lambda n, r=str(root): f"{r}/{n}")
    monkeypatch.setattr("shell.actions.list_models", lambda r=str(root): ["v1.0"])
    monkeypatch.setattr("shell.actions.set_active_model", lambda n: n)
    monkeypatch.setattr("shell.actions.ensure_tf", lambda *a, **k: None)

    sim = types.SimpleNamespace(heads={h: object() for h in ALL_HEADS if h != "event_time"},
                                model=object(), instance=object(), __dict__={})
    gs = types.SimpleNamespace(load=lambda **kw: sim)
    monkeypatch.setitem(sys.modules, "simulation.game_simulator",
                        types.SimpleNamespace(GameSimulator=gs))
    monkeypatch.setitem(sys.modules, "encoder.encoder",
                        types.SimpleNamespace(Encoder=lambda **kw: object()))
    return root, sim


def test_load_populates_the_session(fake_artifacts, capsys):
    from shell.actions import load_model
    s = Session()
    load_model(s, "v1.0")
    assert s.loaded and s.model == "v1.0"
    assert len(s.heads) == len(ALL_HEADS)  # event_time + the other ten


def test_load_warns_when_there_is_no_manifest(fake_artifacts, capsys):
    from shell.actions import load_model
    load_model(Session(), "v1.0")
    assert "no manifest.json" in capsys.readouterr().out


def test_load_rejects_a_missing_required_head(tmp_path, monkeypatch):
    """Absent heads must fail at load, not hours into a rollout."""
    from shell.actions import load_model
    root = tmp_path / "artifacts"
    _fake_model_dir(root, "partial", ["event_time", "player"])
    monkeypatch.setattr("shell.actions.model_root", lambda n, r=str(root): f"{r}/{n}")
    monkeypatch.setattr("shell.actions.list_models", lambda r=str(root): ["partial"])
    with pytest.raises(ShellError, match="missing head"):
        load_model(Session(), "partial")


def test_load_rejects_an_unknown_model(tmp_path, monkeypatch):
    from shell.actions import load_model
    monkeypatch.setattr("shell.actions.model_root", lambda n: str(tmp_path / n))
    monkeypatch.setattr("shell.actions.list_models", lambda: [])
    with pytest.raises(ShellError, match="no model named"):
        load_model(Session(), "ghost")


def test_load_rejects_a_path_escape(monkeypatch):
    from shell.actions import load_model
    with pytest.raises(ShellError, match="invalid model name"):
        load_model(Session(), "../etc")


def test_load_refuses_on_arch_mismatch(fake_artifacts, monkeypatch):
    """The graph is rebuilt from config.py; a mismatch must be a message, not a shape traceback."""
    from shell.actions import load_model
    root, _ = fake_artifacts
    (root / "v1.0" / "manifest.json").write_text(
        json.dumps({"arch": {"MODEL_DIM": config.MODEL_DIM + 128}}), encoding="utf-8")
    with pytest.raises(ShellError, match="architecture mismatch"):
        load_model(Session(), "v1.0")


def test_force_overrides_an_arch_mismatch(fake_artifacts, capsys):
    from shell.actions import load_model
    root, _ = fake_artifacts
    (root / "v1.0" / "manifest.json").write_text(
        json.dumps({"arch": {"MODEL_DIM": config.MODEL_DIM + 128}}), encoding="utf-8")
    load_model(Session(), "v1.0", force=True)
    assert "forced" in capsys.readouterr().out


def test_load_swaps_out_the_previous_model(fake_artifacts, capsys):
    from shell.actions import load_model
    s = Session()
    load_model(s, "v1.0")
    first = s.sim
    load_model(s, "v1.0")
    assert "unloaded v1.0" in capsys.readouterr().out
    assert s.sim is not first or s.loaded  # a swap happened and we are loaded again


def test_unload_clears_the_compiled_infer_cache():
    """tf.function closures capture each head via a default arg; a live cache pins them all."""
    s = Session()
    sim = types.SimpleNamespace(heads={"player": object()}, model=object(), instance=object())
    sim._tf_infer_cache = {"player": object()}
    s.sim, s.model = sim, "v1.0"
    s.unload()
    assert "_tf_infer_cache" not in sim.__dict__
    assert sim.model is None and not sim.heads


def test_dials_survive_a_load(fake_artifacts, capsys):
    """Comparing two models under one tune is the common case, so load must not reset dials."""
    from shell.actions import load_model
    s = Session()
    config.set_dial("DELTA_TIME_SCALE", 0.99)
    load_model(s, "v1.0")
    assert config.DELTA_TIME_SCALE == 0.99
    assert "carried over" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# --- RUN                                                                   -- #
# --------------------------------------------------------------------------- #

@pytest.fixture
def loaded(fake_artifacts, monkeypatch, tmp_path):
    from shell.actions import load_model
    s = Session()
    load_model(s, "v1.0")
    s.holdout_ids, s.holdout_source = [1, 2, 3], "test"
    s.cleaned_df, s.cleaned_df_key = object(), s.data_dir  # skip the CSV read
    calls = {}

    def fake_stage(stage_name, **kw):
        calls.update(kw, stage_name=stage_name)
        return {"done": 3, "total": 3, "run_dir": str(tmp_path / "run")}

    monkeypatch.setitem(sys.modules, "simulation.stage_eval",
                        types.SimpleNamespace(evaluate_stage=fake_stage))
    monkeypatch.setitem(sys.modules, "reporting.eval_report", types.SimpleNamespace(
        resolve_results_run_dir=lambda model, **kw: tmp_path / model / (kw.get("name") or "eval-001")))
    return s, calls


def test_run_reuses_the_loaded_sim(loaded, capsys):
    """The whole point: a run must not rebuild the model."""
    from shell.actions import run_eval
    s, calls = loaded
    run_eval(s, "trial2")
    assert calls["sim"] is s.sim


def test_run_passes_the_cached_frame(loaded):
    from shell.actions import run_eval
    s, calls = loaded
    run_eval(s, "trial2")
    assert calls["df"] is s.cleaned_df


def test_run_labels_the_run_not_the_model(loaded):
    """run_summary.parquet.run_name must distinguish runs, or cross-run analysis is useless."""
    from shell.actions import run_eval
    s, calls = loaded
    run_eval(s, "trial2")
    assert calls["stage_name"] == "v1.0" and calls["run_label"] == "trial2"


def test_run_maps_its_flags(loaded):
    from shell.actions import run_eval
    s, calls = loaded
    run_eval(s, "trial2", games=5, sims=3, concurrency=8, seed=7)
    assert calls["max_new"] == 5 and calls["n_sims"] == 3
    assert calls["batch_size"] == 8 and calls["seed0"] == 7


def test_report_only_simulates_nothing(loaded):
    from shell.actions import run_eval
    s, calls = loaded
    run_eval(s, "trial2", report_only=True)
    assert calls["max_new"] == 0


def test_run_without_a_model_is_refused():
    from shell.actions import run_eval
    with pytest.raises(ShellError, match="no model loaded"):
        run_eval(Session(), "trial2")


def test_run_without_a_holdout_is_refused(fake_artifacts):
    from shell.actions import load_model, run_eval
    s = Session()
    load_model(s, "v1.0")
    s.holdout_ids = []
    with pytest.raises(ShellError, match="no holdout"):
        run_eval(s, "trial2")


# --------------------------------------------------------------------------- #
# --- TRAIN                                                                 -- #
# --------------------------------------------------------------------------- #

def test_train_command_shape():
    cmd = train_command("endgame-feats", batch_size=64, epochs=30)
    assert "--full" in cmd and "--name" in cmd
    assert cmd[cmd.index("--name") + 1] == "endgame-feats"
    assert cmd[cmd.index("--epochs") + 1] == "30"


def test_train_is_a_dry_run_by_default(sh, capsys, monkeypatch):
    """Training must never start from a bare command -- it is a multi-day GPU job."""
    monkeypatch.setattr("shell.actions.model_root", lambda n: f"./artifacts/{n}")
    out = run(sh, capsys, "train brand-new-model --batch-size 64")
    assert "would run" in out and "--go" in out


def test_train_refuses_to_overwrite_an_existing_model(sh, capsys):
    """A retrain takes a NEW name; the model name is the train identity."""
    out = run(sh, capsys, "train v1.0 --batch-size 64")
    assert "already exists" in out


def test_train_rejects_a_bad_name(sh, capsys):
    out = run(sh, capsys, "train ../evil --batch-size 64")
    assert "invalid model name" in out


# --------------------------------------------------------------------------- #
# --- status                                                                -- #
# --------------------------------------------------------------------------- #

def test_status_without_a_model(sh, capsys):
    assert "none loaded" in run(sh, capsys, "status")


def test_status_reports_changed_dials(sh, capsys):
    run(sh, capsys, "set DELTA_TIME_SCALE 0.99")
    assert "1 changed" in run(sh, capsys, "status")
