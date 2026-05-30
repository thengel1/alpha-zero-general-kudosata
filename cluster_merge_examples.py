import os
import pickle
import argparse
from collections import deque


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", type=str, default="./temp/cluster_selfplay")
    parser.add_argument("--output-file", type=str, default="./temp/cluster_train_examples.pkl")
    parser.add_argument("--maxlen", type=int, default=200000)

    args = parser.parse_args()

    merged_examples = deque([], maxlen=args.maxlen)
    all_infos = []

    files = [
        f for f in os.listdir(args.input_dir)
        if f.startswith("examples_worker_") and f.endswith(".pkl")
    ]

    files.sort()

    for filename in files:
        path = os.path.join(args.input_dir, filename)

        with open(path, "rb") as f:
            data = pickle.load(f)

        merged_examples.extend(data["examples"])
        all_infos.extend(data["infos"])

        print(f"Loaded {filename}: {len(data['examples'])} examples")

    with open(args.output_file, "wb") as f:
        pickle.dump({
            "examples": list(merged_examples),
            "infos": all_infos
        }, f)

    print(f"Merged {len(merged_examples)} examples")
    print(f"Saved to {args.output_file}")


if __name__ == "__main__":
    main()