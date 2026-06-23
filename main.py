import argparse

# Import TensorFlow before pandas-using project modules. On Windows, importing
# pandas first can break TF's native DLL initialization ("DLL load failed while
# importing _pywrap_tensorflow_internal"). Loading TF first makes it robust.
try:  # noqa: SIM105
    import tensorflow  # noqa: F401
except Exception:
    pass

from config import HOLDOUT_FRAC
from data_cleaner import DataCleaner
from encoder.encoder import Encoder
from models.pipeline import load_all, run_all
from models.registry import MODEL_REGISTRY

# "all" runs the full dependency-ordered sequence (event_time -> player -> conditional
# heads) via models.pipeline; individual keys train just that model.
MODEL_CHOICES = sorted(MODEL_REGISTRY) + ["all"]


def run_prediction(args):
    """Load trained models, simulate one game, and write CSV + box-score artifacts."""
    import json

    from simulation.game_input import GameInput
    from simulation.predict_game import predict_from_game, predict_game

    common = dict(home_team=args.home_team, away_team=args.away_team,
                  possession=args.possession, seed=args.seed, greedy=args.greedy)

    if args.from_game is not None:
        predict_from_game(args.from_game, data_dir=args.data_dir,
                          use_model_starters=args.model_starters, **common)
    elif args.spec is not None:
        spec = GameInput.from_dict(json.loads(open(args.spec, encoding="utf-8").read()))
        predict_game(spec, **common)
    else:
        raise SystemExit("--predict requires either --from-game <id> or --spec <file>.")


def main():
    parser = argparse.ArgumentParser(
        description="CourtVisionIQ pipeline: clean -> preprocess -> train a registered model (or 'all')."
    )
    parser.add_argument("--model", default="event_time", choices=MODEL_CHOICES,
                        help="Which model to preprocess/train, or 'all' for the full "
                             "sequence (default: event_time).")
    parser.add_argument("--clean", action="store_true",
                        help="Re-run raw play-by-play cleaning into ./data first.")
    parser.add_argument("--clean-start", type=int, default=0,
                        help="First raw-file index to clean (with --clean).")
    parser.add_argument("--clean-end", type=int, default=None,
                        help="Last raw-file index to clean (with --clean).")
    parser.add_argument("--raw-dir", default=None,
                        help="Directory of raw master files to clean (default: ./RawData/MasterFiles).")
    parser.add_argument("--data-dir", default="./data",
                        help="Directory of cleaned season CSVs.")
    parser.add_argument("--train", action="store_true",
                        help="Train the model after preprocessing.")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing step (useful when cleaning only).")
    parser.add_argument("--rebuild-vocabs", action="store_true",
                        help="Rebuild vocabs from data instead of loading from disk.")
    parser.add_argument("--holdout-frac", type=float, default=HOLDOUT_FRAC,
                        help="Fraction of games fully reserved as a real-game holdout "
                             "(never trained on or used for early stopping).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-report", action="store_true",
                        help="Disable the standardized training/testing report.")
    parser.add_argument("--run-name", default=None,
                        help="Optional human label folded into the report run id.")
    parser.add_argument("--load", action="store_true",
                        help="After any training, load all trained models (ModelBundle) "
                             "and print which keys came back — a quick artifact smoke test.")

    # --- Single-game prediction (inference rollout) ---
    parser.add_argument("--predict", action="store_true",
                        help="Run ONE game prediction and write a play-by-play CSV + box "
                             "scores under artifacts/predictions/<run_id>/.")
    parser.add_argument("--from-game", type=int, default=None,
                        help="With --predict: pull the matchup spec from a real cleaned game "
                             "(rosters/season/playoff) by globally-unique game_id.")
    parser.add_argument("--spec", default=None,
                        help="With --predict: path to a GameInput JSON "
                             "({home_roster, away_roster, season, playoff}).")
    parser.add_argument("--home-team", default="HOME", help="Home team label for the box score.")
    parser.add_argument("--away-team", default="AWAY", help="Away team label for the box score.")
    parser.add_argument("--possession", default="home", choices=["home", "away"],
                        help="Which team starts with the ball (default: home).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed the rollout RNG for a reproducible game.")
    parser.add_argument("--greedy", action="store_true",
                        help="Argmax instead of sampling (deterministic, for debugging).")
    parser.add_argument("--model-starters", action="store_true",
                        help="With --from-game: choose the opening five with the "
                             "SubstitutionModel instead of seeding the real game's actual "
                             "starters (the default).")
    args = parser.parse_args()

    # Prediction short-circuits the train/clean pipeline.
    if args.predict:
        run_prediction(args)
        return

    # 1) Clean raw data into season CSVs (optional; expensive), then enrich with the
    #    season-level context (per-player rest, team rest, games-played) the models consume.
    #    Enrichment must run before any preprocess so the new columns are present.
    if args.clean:
        DataCleaner(
            start=args.clean_start, end=args.clean_end, data_path=args.raw_dir,
        ).run()
        from season_context import enrich
        enrich(args.data_dir)

    # 2 + 3) Preprocess + train. "all" runs the full dependency-ordered sequence
    # (event_time rebuilds + freezes the shared vocab first, then player, then the
    # conditional heads); a single key trains just that model.
    if args.model == "all":
        run_all(
            data_dir=args.data_dir,
            holdout_frac=args.holdout_frac,
            epochs=args.epochs,
            batch_size=args.batch_size,
            report=not args.no_report,
            run_name=args.run_name,
            skip_preprocess=args.skip_preprocess,
            train=args.train,
        )
    else:
        # Independent stages: clean / preprocess / train can each run on their own.
        # train() loads vocabs + processed tensors from disk, so --skip-preprocess
        # --train trains on already-preprocessed data without redoing preprocessing.
        encoder = Encoder()
        model = MODEL_REGISTRY[args.model](encoder, path=args.data_dir)
        if not args.skip_preprocess:
            model.preprocess(rebuild_vocabs=args.rebuild_vocabs, holdout_frac=args.holdout_frac)
        if args.train:
            model.train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                report=not args.no_report,
                run_name=args.run_name,
            )

    # Optional: load everything back as a smoke test of the saved artifacts.
    if args.load:
        load_all()


if __name__ == "__main__":
    main()
