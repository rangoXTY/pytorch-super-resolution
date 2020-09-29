import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5//2)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=5//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))

        return out