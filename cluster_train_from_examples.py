import os
import pickle
import argparse
import numpy as np

from utils import dotdict
from MCTS import MCTS
from Arena import Arena
from kudosata.KudosataGame import KudosataGame as Game
from kudosata.NNetWrapper import NNetWrapper as NNet


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--examples-file", type=str, default="./temp/cluster_train_examples.pkl")
    parser.add_argument("--checkpoint", type=str, default="./temp")
    parser.add_argument("--previous", type=str, default="selfplay_current.pth.tar")
    parser.add_argument("--output", type=str, default="best.pth.tar")

    args_cli = parser.parse_args()

    args = dotdict({
        "numIters": 1,
        "numEps": 0,
        "tempThreshold": 20,
        "updateThreshold": 0.50,
        "maxlenOfQueue": 200000,
        "numMCTSSims": 100,
        "arenaCompare": 30,
        "cpuct": 1.75,
        "maxEpisodeSteps": 90,
        "checkpoint": args_cli.checkpoint,
        "load_model": True,
        "load_folder_file": (args_cli.checkpoint, args_cli.previous),
        "numItersForTrainExamplesHistory": 5,
    })

    game = Game()

    nnet = NNet(game)
    nnet.load_checkpoint(args_cli.checkpoint, args_cli.previous)

    pnet = NNet(game)
    pnet.load_checkpoint(args_cli.checkpoint, args_cli.previous)

    with open(args_cli.examples_file, "rb") as f:
        data = pickle.load(f)

    train_examples = data["examples"]

    print("Training examples:", len(train_examples))

    nnet.train(train_examples)

    pmcts = MCTS(game, pnet, args)
    nmcts = MCTS(game, nnet, args)

    arena = Arena(
        lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
        lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
        game
    )

    pwins, nwins, draws = arena.playGames(args.arenaCompare)

    print("NEW/PREV WINS:", nwins, "/", pwins, "DRAWS:", draws)

    if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < args.updateThreshold:
        print("REJECTING NEW MODEL")
    else:
        print("ACCEPTING NEW MODEL")
        nnet.save_checkpoint(args_cli.checkpoint, args_cli.output)


if __name__ == "__main__":
    main()