import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import MAX_SEQUENCE_LENGTH, ROSTER_SIZE, NORM_STATS_PATH
from encoder.encoder import Encoder

# Cleaned-data columns the model consumes.
CATEGORICAL_FIELDS = ["event", "player", "type", "result", "season"]
ROSTER_COLS = {"home_roster": "roster_home", "away_roster": "roster_away"}


class EventTimeModel:
    """
    Core Event/Time Transformer.

    preprocess() turns cleaned, game-grouped CSV(s) into padded, normalized tensor
    arrays (one game = one fixed-length sequence) plus the shared vocab "language"
    and time-normalization stats. model() builds the causal transformer; train()
    fits it. Encoding is standardized so every downstream model speaks the same
    token language.
    """

    def __init__(self, encoder: Encoder,
                 sequence_length=MAX_SEQUENCE_LENGTH,
                 model_dim=256,
                 event_classes=7,
                 path="./data",
                 processed_dir="./data/processed"):

        self.sequence_length = sequence_length
        self.model_dim = model_dim
        self.event_classes = event_classes
        self.encoder = encoder

        self.data_dir = Path(path)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir.resolve()}")
        self.processed_dir = Path(processed_dir)

        self.roster_encoder = None
        self.norm_stats = None

    # =====================
    # --- Data Loading  ---
    # =====================

    def _cleaned_csvs(self):
        """Cleaned season files only (have game_id + roster columns)."""
        out = []
        for p in sorted(self.data_dir.glob("*.csv")):
            try:
                cols = pd.read_csv(p, nrows=0).columns
            except Exception:
                continue
            if "game_id" in cols and "roster_home" in cols:
                out.append(p)
        return out

    def _load_all(self) -> pd.DataFrame:
        frames = []
        offset = 0
        for p in self._cleaned_csvs():
            df = pd.read_csv(p)
            # Keep game_id globally unique across files.
            df["game_id"] = df["game_id"].astype(int) + offset
            offset = int(df["game_id"].max()) + 1
            frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No cleaned CSVs found in {self.data_dir.resolve()}")
        return pd.concat(frames, ignore_index=True)

    # =====================
    # --- Preprocessing ---
    # =====================

    def preprocess(self, rebuild_vocabs=True, test_frac=0.2, seed=42):
        """
        Build the model-ready tensor arrays and persist them, the vocabs, and the
        time-normalization stats. Returns (train, test) dicts of numpy arrays.
        """
        df = self._load_all()

        # 1) Build (or load) the shared vocab language, then FREEZE it so token IDs
        #    are stable and unseen values map to UNK rather than mutating the space.
        if rebuild_vocabs:
            for col, src in ROSTER_COLS.items():
                df[src].apply(self.encoder.encode_roster)
            for field in CATEGORICAL_FIELDS:
                df[field].apply(getattr(self.encoder, f"encode_{field}"))
            self.encoder.save_all()
        else:
            self.encoder.load_all()
        self.encoder.freeze_all()

        # 2) Encode every categorical column to ints and rosters to fixed-5 arrays.
        enc = {f: df[f].apply(getattr(self.encoder, f"encode_{f}")).to_numpy() for f in CATEGORICAL_FIELDS}
        rosters = {
            name: np.stack(df[src].apply(self.encoder.encode_roster).to_numpy())
            for name, src in ROSTER_COLS.items()
        }  # each (N, 5)
        time = df["time"].to_numpy(dtype=np.float64)
        game_id = df["game_id"].to_numpy()

        # 3) Per-game delta_time (no cross-game leakage): first row of each game
        #    gets delta 0; clip guards against any backwards time stamps.
        delta = (
            df.groupby("game_id")["time"].diff().fillna(0).clip(lower=0).to_numpy(dtype=np.float64)
        )

        # 4) Split games into train/test BEFORE computing normalization stats.
        unique_games = np.unique(game_id)
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_games)
        n_test = max(1, int(round(len(unique_games) * test_frac)))
        test_games = set(unique_games[:n_test].tolist())
        train_games = set(unique_games[n_test:].tolist())

        train_mask = np.array([g in train_games for g in game_id])

        # 5) Normalization stats from TRAIN ONLY. time_abs = time/max_time;
        #    delta standardized. Persist so inference uses identical transforms.
        max_time = float(time[train_mask].max()) or 1.0
        train_delta = delta[train_mask]
        delta_mean = float(train_delta.mean())
        delta_std = float(train_delta.std()) or 1.0
        self.norm_stats = {"max_time": max_time, "delta_mean": delta_mean, "delta_std": delta_std}

        time_abs = (time / max_time).astype(np.float32)
        delta_norm = ((delta - delta_mean) / delta_std).astype(np.float32)

        # 6) Assemble per-game padded sequences for each split.
        cols = {
            **{f: enc[f] for f in CATEGORICAL_FIELDS},
            "home_roster": rosters["home_roster"],
            "away_roster": rosters["away_roster"],
            "time_abs": time_abs,
            "delta_time": delta_norm,
        }
        train = self._build_split(cols, game_id, train_games)
        test = self._build_split(cols, game_id, test_games)

        # 7) Persist.
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.processed_dir / "train.npz", **train)
        np.savez_compressed(self.processed_dir / "test.npz", **test)
        Path(NORM_STATS_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(NORM_STATS_PATH).write_text(json.dumps(self.norm_stats, indent=2), encoding="utf-8")

        print(f"Preprocessed {len(train_games)} train / {len(test_games)} test games "
              f"-> {self.processed_dir} (SEQ={self.sequence_length})")
        return train, test

    def _build_split(self, cols, game_id, games) -> dict:
        """
        Pad/truncate each game to sequence_length and stack into batched arrays.

        Produces inputs, shifted targets, and two masks:
          - pad_mask:  1 for real timesteps (incl. the terminal 'end' row), else 0.
                       Used to mask attention over padded timesteps.
          - loss_mask: 1 for real timesteps that have a valid next-step target
                       (i.e. not the last real row), else 0. Used as sample_weight.
        """
        SEQ = self.sequence_length
        PAD_PLAYER = self.encoder.encode_player("PAD")
        PAD_EVENT = self.encoder.encode_event("PAD")
        PAD_TYPE = self.encoder.encode_type("PAD")
        PAD_RESULT = self.encoder.encode_result("PAD")
        PAD_SEASON = self.encoder.encode_season("PAD")
        pad_scalar = {
            "event": PAD_EVENT, "player": PAD_PLAYER, "type": PAD_TYPE,
            "result": PAD_RESULT, "season": PAD_SEASON,
        }

        keys_1d = CATEGORICAL_FIELDS
        keys_roster = ["home_roster", "away_roster"]
        keys_cont = ["time_abs", "delta_time"]

        batches = {k: [] for k in (*keys_1d, *keys_roster, *keys_cont,
                                   "event_target", "time_target", "pad_mask", "loss_mask")}

        game_ids_sorted = [g for g in np.unique(game_id) if g in games]
        for g in game_ids_sorted:
            idx = np.where(game_id == g)[0]
            idx = idx[:SEQ]  # truncate overlong games
            n = len(idx)

            def pad1d(arr, pad_val, dtype):
                buf = np.full((SEQ,), pad_val, dtype=dtype)
                buf[:n] = arr[idx]
                return buf

            for k in keys_1d:
                batches[k].append(pad1d(cols[k], pad_scalar[k], np.int32))

            for k in keys_roster:
                buf = np.full((SEQ, ROSTER_SIZE), PAD_PLAYER, dtype=np.int32)
                buf[:n] = cols[k][idx]
                batches[k].append(buf)

            for k in keys_cont:
                buf = np.zeros((SEQ, 1), dtype=np.float32)
                buf[:n, 0] = cols[k][idx]
                batches[k].append(buf)

            # Targets: next-step shift within this game.
            event_t = np.full((SEQ,), PAD_EVENT, dtype=np.int32)
            time_t = np.zeros((SEQ, 1), dtype=np.float32)
            if n > 1:
                event_t[: n - 1] = cols["event"][idx][1:]
                time_t[: n - 1, 0] = cols["delta_time"][idx][1:]
            batches["event_target"].append(event_t)
            batches["time_target"].append(time_t)

            pad_mask = np.zeros((SEQ,), dtype=np.float32)
            pad_mask[:n] = 1.0
            loss_mask = np.zeros((SEQ,), dtype=np.float32)
            loss_mask[: max(n - 1, 0)] = 1.0  # last real row has no next target
            batches["pad_mask"].append(pad_mask)
            batches["loss_mask"].append(loss_mask)

        return {k: np.stack(v) if v else np.empty((0,)) for k, v in batches.items()}

    # =====================
    # --- Model / Train ---
    # =====================

    def model(self):
        """Built in Phase 3."""
        raise NotImplementedError("EventTimeModel.model() not implemented yet")

    def train(self):
        """Built in Phase 4."""
        raise NotImplementedError("EventTimeModel.train() not implemented yet")
