import os
import torch
import torch.optim as optim
from .KudosataNNet import KudosataNNet as nnet
from NeuralNet import NeuralNet


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.args = type('Args', (), {'lr': 0.001, 'dropout': 0.3, 'epochs': 10,
                                      'batch_size': 64, 'cuda': torch.cuda.is_available(),
                                      'num_channels': 256, 'depth': 10})()
        self.nnet = nnet(game, self.args)
        if self.args.cuda:
            self.nnet.cuda()

    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters())

        for epoch in range(self.args.epochs):
            self.nnet.train()

    def predict(self, board):
        board = torch.FloatTensor(board.astype(float))
        if self.args.cuda: board = board.contiguous().cuda()
        board = board.view(1, 33, self.nnet.board_x, self.nnet.board_y)

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        checkpoint = torch.load(filepath)
        self.nnet.load_checkpoint(checkpoint['state_dict'])