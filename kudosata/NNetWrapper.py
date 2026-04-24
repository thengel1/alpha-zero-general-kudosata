import os
import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from .KudosataNNet import KudosataNNet as nnet
from NeuralNet import NeuralNet


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.args = type('Args', (), {'lr': 0.001, 'dropout': 0.3,'epochs': 1,
                                      'batch_size': 32, 'cuda': torch.cuda.is_available(),
                                      'num_channels': 64, 'depth': 4})()
        self.nnet = nnet(game, self.args)
        if self.args.cuda:
            self.nnet.cuda()


    def train(self, examples):
        print("TRAIN START")
        print("nb examples =", len(examples))

        if len(examples) == 0:
            print("NO EXAMPLES, SKIP TRAIN")
            return

        optimizer = optim.Adam(self.nnet.parameters(), lr=self.args.lr)

        for epoch in range(self.args.epochs):
            print(f"EPOCH {epoch + 1}/{self.args.epochs}")
            self.nnet.train()
            random.shuffle(examples)

            batch_count = int(np.ceil(len(examples) / self.args.batch_size))
            print("batch_count =", batch_count)

            for batch_idx in range(batch_count):
                print(f"batch {batch_idx + 1}/{batch_count}")

                batch_examples = examples[
                                 batch_idx * self.args.batch_size:(batch_idx + 1) * self.args.batch_size
                                 ]

                boards, target_pis, target_vs = list(zip(*batch_examples))

                boards = torch.FloatTensor(np.array(boards, dtype=np.float32))
                target_pis = torch.FloatTensor(np.array(target_pis, dtype=np.float32))
                target_vs = torch.FloatTensor(np.array(target_vs, dtype=np.float32))

                if self.args.cuda:
                    boards = boards.contiguous().cuda()
                    target_pis = target_pis.contiguous().cuda()
                    target_vs = target_vs.contiguous().cuda()

                out_pi, out_v = self.nnet(boards)

                loss_pi = -torch.sum(target_pis * out_pi) / target_pis.size(0)
                loss_v = F.mse_loss(out_v.view(-1), target_vs.view(-1))
                loss = loss_pi + loss_v

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print("loss =", float(loss.item()))

        print("TRAIN END")

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float32))
        if self.args.cuda: board = board.contiguous().cuda()
        board = board.view(1, 41, self.nnet.board_x, self.nnet.board_y)

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.item()

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        checkpoint = torch.load(filepath,map_location='cuda' if self.args.cuda else 'cpu')
        self.nnet.load_state_dict(checkpoint['state_dict'])