import torch
import torch.nn as nn
import torch.nn.functional as F


class KudosataNNet(nn.Module):
    def __init__(self, game, args):
        self.board_x, self.board_y = game.getBoardSize()[1:]
        self.action_size = game.getActionSize()
        self.args = args

        super(KudosataNNet, self).__init__()

        self.conv1 = nn.Conv2d(33, args.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

        self.res_layers = nn.ModuleList([
            ResidualBlock(args.num_channels) for _ in range(args.depth)
        ])

        self.policy_conv = nn.Conv2d(args.num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.board_x * self.board_y, self.action_size)

        self.value_conv = nn.Conv2d(args.num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * self.board_x * self.board_y, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, s):
        s = F.relu(self.bn1(self.conv1(s)))

        for layer in self.res_layers:
            s = layer(s)

        p = F.relu(self.policy_bn(self.policy_conv(s)))
        p = p.view(-1, 2 * self.board_x * self.board_y)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)

        v = F.relu(self.value_bn(self.value_conv(s)))
        v = v.view(-1, 1 * self.board_x * self.board_y)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)