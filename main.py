import argparse

# Import TensorFlow before pandas-using project modules. On Windows, importing
# pandas first can break TF's native DLL initialization ("DLL load failed while
# importing _pywrap_tensorflow_internal"). Loading TF first makes it robust.
try:  # noqa: SIM105
    import tensorflow  # noqa: F401
except Exception:
    pass

from data_cleaner import DataCleaner
from encoder.encoder import Encoder
from models.event_time_model import EventTimeModel


def main():
    parser = argparse.ArgumentParser(
        description="CourtVisionIQ pipeline: clean -> preprocess -> train the Event/Time model."
    )
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
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-report", action="store_true",
                        help="Disable the standardized training/testing report.")
    parser.add_argument("--run-name", default=None,
                        help="Optional human label folded into the report run id.")
    args = parser.parse_args()

    # 1) Clean raw data into season CSVs (optional; expensive).
    if args.clean:
        DataCleaner(
            start=args.clean_start, end=args.clean_end, data_path=args.raw_dir,
        ).run()

    # Independent stages: clean / preprocess / train can each run on their own.
    # train() loads vocabs + processed tensors from disk, so --skip-preprocess
    # --train trains on already-preprocessed data without redoing preprocessing.
    encoder = Encoder()
    model = EventTimeModel(encoder, path=args.data_dir)

    # 2) Build the shared encoding "language" + model-ready tensors.
    if not args.skip_preprocess:
        model.preprocess(rebuild_vocabs=args.rebuild_vocabs)

    # 3) Train the Event/Time transformer (if --train flag is provided).
    if args.train:
        model.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            report=not args.no_report,
            run_name=args.run_name,
        )


if __name__ == "__main__":
    main()
