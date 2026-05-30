import os
import argparse
import pickle
import torch

from utils import dotdict
from MCTS import MCTS
from Coach import Coach
from kudosata.KudosataGame import KudosataGame as Game
from kudosata.NNetWrapper import NNetWrapper as NNet


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--num-eps", type=int, required=True)
    parser.add_argument("--checkpoint", type=str, default="./temp")
    parser.add_argument("--checkpoint-file", type=str, default="selfplay_current.pth.tar")
    parser.add_argument("--output-dir", type=str, default="./temp/cluster_selfplay")

    args_cli = parser.parse_args()

    torch.set_num_threads(1)

    args = dotdict({
        "numIters": 1,
        "numEps": args_cli.num_eps,
        "tempThreshold": 20,
        "updateThreshold": 0.50,
        "maxlenOfQueue": 200000,
        "numMCTSSims": 100,
        "arenaCompare": 30,
        "cpuct": 1.75,
        "maxEpisodeSteps": 90,
        "checkpoint": args_cli.checkpoint,
        "load_model": True,
        "load_folder_file": (args_cli.checkpoint, args_cli.checkpoint_file),
        "numItersForTrainExamplesHistory": 5,
    })

    args.disable_visual = True

    os.makedirs(args_cli.output_dir, exist_ok=True)

    game = Game()
    nnet = NNet(game)
    nnet.load_checkpoint(args_cli.checkpoint, args_cli.checkpoint_file)

    coach = Coach(game, nnet, args)

    all_examples = []
    infos = []

    for _ in range(args_cli.num_eps):
        coach.mcts = MCTS(game, nnet, args)
        examples, info = coach.executeEpisode()

        all_examples.extend(examples)
        infos.append(info)

    output_file = os.path.join(
        args_cli.output_dir,
        f"examples_worker_{args_cli.worker_id}.pkl"
    )

    with open(output_file, "wb") as f:
        pickle.dump({
            "examples": all_examples,
            "infos": infos,
            "worker_id": args_cli.worker_id
        }, f)

    print(f"Worker {args_cli.worker_id} saved {len(all_examples)} examples to {output_file}")


if __name__ == "__main__":
    main()