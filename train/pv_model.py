import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class PVModel(nn.Module):
    """
    CNN-based Policy-Value network.
    Input: (batch_size, 5, N, N)
    Outputs:
      - policy: logits over 4*max_steps actions (softmaxed)
      - value: scalar in [-1,1]
    """
    def __init__(self, board_size=20, in_channels=5, num_res_blocks=5, filter_size=64, max_steps=20):
        super(PVModel, self).__init__()
        self.board_size = board_size
        self.max_steps = max_steps

        # Initial conv
        self.conv_pre = nn.Sequential(
            nn.Conv2d(in_channels, filter_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(filter_size),
            nn.ReLU(),
        )
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(filter_size) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(filter_size, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.policy_fc = nn.Linear(2 * board_size * board_size, 4 * max_steps)

        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(filter_size, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(board_size * board_size, filter_size),
            nn.ReLU(),
            nn.Linear(filter_size, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Shared trunk
        x = self.conv_pre(x)
        x = self.res_blocks(x)

        # Policy
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        p = torch.softmax(self.policy_fc(p), dim=1)

        # Value
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = self.value_fc(v)

        return p, v.squeeze(-1)
