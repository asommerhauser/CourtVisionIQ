import argparse

from data_cleaner import DataCleaner
from encoder.encoder import Encoder
from models.event_time_model import EventTimeModel


def main():
    parser = argparse.ArgumentParser(
        description="CourtVisionIQ pipeline: clean -> preprocess -> train the Event/Time model."
    )
    parser.add_argument("--clean", action="store_true",
                        help="Re-run raw play-by-play cleaning into ./Data first.")
    parser.add_argument("--clean-start", type=int, default=0,
                        help="First raw-file index to clean (with --clean).")
    parser.add_argument("--clean-end", type=int, default=None,
                        help="Last raw-file index to clean (with --clean).")
    parser.add_argument("--data-dir", default="./Data",
                        help="Directory of cleaned season CSVs.")
    parser.add_argument("--no-train", action="store_true",
                        help="Stop after preprocessing (skip training).")
    parser.add_argument("--rebuild-vocabs", action="store_true",
                        help="Rebuild vocabs from data instead of loading from disk.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # 1) Clean raw data into season CSVs (optional; expensive).
    if args.clean:
        DataCleaner(start=args.clean_start, end=args.clean_end).run()

    # 2) Build the shared encoding "language" + model-ready tensors.
    encoder = Encoder()
    model = EventTimeModel(encoder, path=args.data_dir)
    model.preprocess(rebuild_vocabs=args.rebuild_vocabs)

    # 3) Train the Event/Time transformer.
    if not args.no_train:
        model.train(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
