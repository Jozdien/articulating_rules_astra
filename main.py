import argparse
from src.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default="configs/experiment.yaml")
    parser.add_argument("--run-dir", help="Existing run directory to use")
    args = parser.parse_args()

    runner = Runner(args.config, run_dir=args.run_dir)
    runner.run()


if __name__ == "__main__":
    main()
