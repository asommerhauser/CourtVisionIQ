"""
Per-model manifest + vocab snapshot: what makes a set of weights self-describing.

Weights alone do not record the architecture that produced them, and the graph is rebuilt from
``config.py`` at load time, so editing a capacity dim turns a later load into an opaque Keras
shape error. The manifest records the arch (among the seed, epochs, git commit and data cut) so
that becomes a readable refusal instead.

The vocab snapshot fixes a sharper problem. Every head's ``save_artifacts()`` calls
``encoder.save_all()``, which writes the *shared* ``encoder/vocabs/``. Training a new model
therefore rewrites the vocabs an older model's weights were built against, and its embedding
tables are sized for the old token count. Copying the vocabs into ``artifacts/<name>/vocabs/`` at
train time pins them to those weights; ``shell.actions.load_model`` prefers that copy when it
exists. The shared directory then degrades to "the vocabs of the most recent train", which is
harmless once every model carries its own.

``python -m models.manifest adopt <name>`` back-fills both for a model trained before this
existed. It requires no retrain.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import shutil
import subprocess
from pathlib import Path

import config
from player_floor import ANON_FILENAME
from models.artifacts import ModelArtifacts, list_models, model_root

MANIFEST_NAME = "manifest.json"
VOCAB_SNAPSHOT_DIR = "vocabs"
SCHEMA = 2

# Capacity dims the graph is rebuilt from; a mismatch means the weights will not load.
#
# LOCAL_ATTENTION_* are here for the opposite reason: they change no weight shapes at all, so a
# model trained with local heads reloads into an all-global graph with no error and no shape
# mismatch -- just quietly wrong attention in every rollout. Recording them is the only thing
# that makes that mismatch visible.
ARCH_KEYS = ("MODEL_DIM", "NUM_LAYERS", "NUM_HEADS", "FF_DIM", "ROSTER_SAB_LAYERS",
             "MAX_SEQUENCE_LENGTH", "ROSTER_SIZE", "BENCH_SIZE",
             "LOCAL_ATTENTION_HEADS", "LOCAL_ATTENTION_WINDOW",
             "REGIME_ENABLED", "REGIME_DIM",
             # 3.2 W6. FILM_ENABLED off produces a graph MISSING layers rather than a graph with
             # differently-shaped ones, which is the LOCAL_ATTENTION_* failure class this list exists
             # for: load_weights matches by name, so the absent ones are skipped in silence.
             "FILM_ENABLED", "FILM_DIM")


def feature_snapshot() -> dict:
    """WHICH INPUTS a model was trained with -- not how big it is.

    ARCH_KEYS records capacity; this records the input signature. 3.0 widened the fusion three
    times in one retrain (the eight team priors, running_pace, the regime latent) and raised the
    roster encoder's per-player scalar count from 4 to 14, and none of that is a capacity dim.

    Most of these changes DO fail loudly at ``load_weights``, because ``fusion_projection``'s kernel
    shape is a function of the concatenated width -- that is why ``build_backbone`` takes a list of
    parts rather than a pre-fused tensor. But not all of them. A change that swaps one game-state
    key for another of the same width reloads with no error at all and means something different on
    every row, which is exactly the failure LOCAL_ATTENTION_* was added to ARCH_KEYS to prevent.
    Recording the names is the only thing that makes it visible.
    """
    from models.game_state_features import GAME_STATE_KEYS
    from models.prior_features import PRIOR_INPUT_KEYS
    from models.regime import REGIME_KEY
    from models.rotation_features import (
        NUM_BENCH_SCALARS, NUM_ROSTER_SCALARS, ROSTER_STATE_KEYS)
    from models.season_features import SEASON_INPUT_KEYS
    from player_priors import PLAYER_PRIOR_KEYS, TEAM_PRIOR_KEYS

    return {
        "game_state_keys": list(GAME_STATE_KEYS),
        "roster_state_keys": list(ROSTER_STATE_KEYS),
        "season_input_keys": list(SEASON_INPUT_KEYS),
        "prior_input_keys": list(PRIOR_INPUT_KEYS),
        "player_prior_keys": list(PLAYER_PRIOR_KEYS),
        "team_prior_keys": list(TEAM_PRIOR_KEYS),
        "regime_key": REGIME_KEY,
        "num_roster_scalars": NUM_ROSTER_SCALARS,
        "num_bench_scalars": NUM_BENCH_SCALARS,
    }


def feature_mismatch(recorded: dict | None) -> list[str]:
    """Which parts of a recorded feature signature disagree with this build. Empty means agree.

    ``None`` or ``{}`` means a manifest written before 3.0, which carries no signature. That is not
    a mismatch -- it is an absence, and it reads as one so a v1.0 or 2.0 bundle stays loadable.
    """
    if not recorded:
        return []
    current = feature_snapshot()
    return [f"{key}: trained with {recorded.get(key)!r}, this build has {current[key]!r}"
            for key in current if key in recorded and recorded[key] != current[key]]


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _sha256(path, limit=None) -> str:
    """Hash a file. ``limit`` caps the bytes read -- weight files are ~250 MB each."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        read = 0
        while chunk := fh.read(1 << 20):
            h.update(chunk)
            read += len(chunk)
            if limit and read >= limit:
                break
    return h.hexdigest()


def vocab_fingerprint(encoder) -> dict:
    """Per-vocab ``{size, sha256}``, sized by token count.

    Size is what actually matters: it is the embedding-table row count baked into the weights.
    """
    out = {}
    for name, vocab in encoder.vocabs.items():
        path = Path(encoder.vocab_dir) / f"{name}_vocab.json"
        entry = {"size": len(vocab.token_to_id) if hasattr(vocab, "token_to_id") else None}
        if path.is_file():
            entry["sha256"] = _sha256(path)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "next_token" in data:
                    entry["size"] = data["next_token"]
                elif isinstance(data, dict) and isinstance(data.get("token_to_id"), dict):
                    entry["size"] = len(data["token_to_id"])
            except (OSError, json.JSONDecodeError):
                pass
        out[name] = entry
    # The alias map decides WHICH NAME each below-floor player is encoded under, so swapping it
    # changes the meaning of every anonymous slot without changing any vocab size. Fingerprinted for
    # the same reason LOCAL_ATTENTION_* is in ARCH_KEYS: a same-shape change that reloads cleanly and
    # means something different on every row.
    anon = Path(encoder.vocab_dir) / ANON_FILENAME
    if anon.is_file():
        try:
            data = json.loads(anon.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        out["anon"] = {"size": data.get("n_slots"), "n_aliased": data.get("n_aliased"),
                       "floor": data.get("floor"), "sha256": _sha256(anon)}
    return out


def snapshot_vocabs(encoder, root) -> Path:
    """Copy the encoder's vocab files into ``<root>/vocabs/``, pinning them to these weights."""
    dest = Path(root) / VOCAB_SNAPSHOT_DIR
    dest.mkdir(parents=True, exist_ok=True)
    src = Path(encoder.vocab_dir)
    for f in sorted(src.glob("*.json")):
        shutil.copy2(f, dest / f.name)
    return dest


def encoder_for(root):
    """An ``Encoder`` bound to the model's own vocab snapshot, or the shared one if absent."""
    from encoder.encoder import Encoder
    snap = Path(root) / VOCAB_SNAPSHOT_DIR
    return Encoder(vocab_dir=snap) if snap.is_dir() else Encoder()


def read_manifest(root) -> dict:
    """The manifest at ``root``, or an empty dict -- pre-manifest models must still load."""
    p = Path(root) / MANIFEST_NAME
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def write_manifest(root, **fields) -> Path:
    """Merge ``fields`` into the manifest at ``root`` and write it. Returns the path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data = read_manifest(root)
    data.update(fields)
    data.setdefault("schema", SCHEMA)
    p = root / MANIFEST_NAME
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return p


def arch_snapshot() -> dict:
    """The capacity dims currently in ``config.py``."""
    return {k: getattr(config, k) for k in ARCH_KEYS if hasattr(config, k)}


def new_manifest(name, *, epochs=None, batch_size=None, seed=None, data_dir=None,
                 processed_dir=None, n_games=None, boundary_idx=None,
                 holdout_game_ids=None, **extra) -> dict:
    """The stub written at train setup, before any head has finished."""
    return {
        "schema": SCHEMA,
        "name": name,
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "platform": platform.platform(),
        "seed": seed if seed is not None else getattr(config, "SEED", None),
        "epochs": epochs,
        "batch_size": batch_size,
        "arch": arch_snapshot(),
        # WHICH inputs, not how many parameters. See feature_snapshot.
        "features": feature_snapshot(),
        "data": {
            "data_dir": data_dir,
            "processed_dir": processed_dir,
            "n_games": n_games,
            "boundary_idx": boundary_idx,
            # WHICH corpus, not just how much of it. Before 3.2 the only corpus provenance was
            # n_games and boundary_idx, so two models trained on different season ranges were
            # indistinguishable from their manifests -- and n_games alone cannot tell "the corpus
            # grew" from "the floor moved".
            "min_train_season": getattr(config, "MIN_TRAIN_SEASON", None),
            "test_frac": getattr(config, "TEST_FRAC", None),
            "holdout_frac": getattr(config, "HOLDOUT_FRAC", None),
            "recency": {
                "weighting": getattr(config, "RECENCY_WEIGHTING", None),
                "halflife_seasons": getattr(config, "RECENCY_HALFLIFE_SEASONS", None),
                "floor": getattr(config, "RECENCY_FLOOR", None),
            },
        },
        "holdout_game_ids": list(holdout_game_ids or []),
        "heads": {},
        "vocabs": {},
        "recommended_dials": config.get_dials(),
        **extra,
    }


def record_head(root, key, *, params=None) -> Path:
    """Append one finished head to the manifest, right after its weights are saved."""
    data = read_manifest(root)
    arts = ModelArtifacts.for_key(key, root)
    heads = data.setdefault("heads", {})
    entry = {"trained_at": _dt.datetime.now().isoformat(timespec="seconds")}
    if arts.weights_path.exists():
        entry["weights_bytes"] = arts.weights_path.stat().st_size
        # Prefix hash only: a full sha256 of a ~250 MB file on every head is not worth the I/O.
        entry["weights_sha256_prefix"] = _sha256(arts.weights_path, limit=1 << 22)
    if params is not None:
        entry["params"] = params
    heads[key] = entry
    return write_manifest(root, heads=heads)


def adopt(name, *, models_root=None, echo=print) -> Path:
    """Back-fill a manifest + vocab snapshot for a model trained before either existed.

    No retrain: the arch is taken from the current ``config.py`` (documented assumption -- if it
    has changed since the model was trained, the recorded arch is wrong and the check it enables
    would be too), timestamps from the weight files, and the holdout from the processed manifest.
    """
    from encoder.encoder import Encoder

    root = Path(model_root(name) if models_root is None else f"{models_root}/{name}")
    if not root.is_dir():
        raise FileNotFoundError(f"no model at {root}. Available: {', '.join(list_models())}")

    heads = {}
    newest = None
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == VOCAB_SNAPSHOT_DIR:
            continue
        arts = ModelArtifacts.for_key(d.name, root)
        if not arts.exists():
            continue
        st = arts.weights_path.stat()
        newest = max(newest or st.st_mtime, st.st_mtime)
        heads[d.name] = {
            "trained_at": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "weights_bytes": st.st_size,
        }
    if not heads:
        raise FileNotFoundError(f"{root} contains no head weights; nothing to adopt.")

    dest = snapshot_vocabs(Encoder(), root)
    echo(f"  vocabs snapshotted -> {dest}")

    holdout = []
    man = Path(config.ROOT_DIR) / "data" / "processed" / config.HOLDOUT_MANIFEST_NAME
    if man.is_file():
        try:
            holdout = [int(g) for g in json.loads(man.read_text(encoding="utf-8"))]
        except (OSError, json.JSONDecodeError, ValueError):
            holdout = []
    if not holdout:
        # The processed manifest is rewritten by preprocess and can be left empty; the ids also
        # survive in any previous run's per-game folder names.
        from shell.session import holdout_from_results
        holdout = holdout_from_results(name)
        if holdout:
            echo(f"  holdout recovered from results/{name}/ ({len(holdout)} games)")

    # The shared encoder/vocabs/norm_stats.json is pipeline-level and is currently overwritten by
    # the test fixtures, so prefer this model's real event_time stats when snapshotting.
    head_stats = Path(root) / "event_time" / "norm_stats.json"
    if head_stats.is_file():
        shutil.copy2(head_stats, dest / "norm_stats.json")
        echo(f"  norm_stats taken from {head_stats} (the shared copy is fixture-polluted)")

    data = new_manifest(name, holdout_game_ids=holdout,
                        data_dir="./data", processed_dir="./data/processed")
    data.update({
        "backfilled": True,
        "backfill_note": ("Adopted from existing weights. arch/seed/recency come from the current "
                          "config.py, not from the original train -- they are an assumption."),
        "created_at": _dt.datetime.fromtimestamp(newest).isoformat(timespec="seconds"),
        "finished_at": _dt.datetime.fromtimestamp(newest).isoformat(timespec="seconds"),
        "epochs": None,
        "batch_size": None,
        "heads": heads,
        "vocabs": vocab_fingerprint(Encoder(vocab_dir=dest)),
    })
    p = write_manifest(root, **data)
    echo(f"  manifest written -> {p}  ({len(heads)} heads, {len(holdout)} holdout games)")
    return p


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m models.manifest",
                                 description="Inspect or back-fill model manifests.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("adopt", help="back-fill a manifest + vocab snapshot (no retrain)")
    a.add_argument("name")
    s = sub.add_parser("show", help="print a model's manifest")
    s.add_argument("name")
    args = ap.parse_args(argv)

    if args.cmd == "adopt":
        adopt(args.name)
    else:
        data = read_manifest(model_root(args.name))
        if not data:
            print(f"{args.name}: no manifest (run 'adopt {args.name}')")
            return 1
        print(json.dumps(data, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
