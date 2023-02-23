import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn(self.conv(x))
        out += self.shortcut(x)
        return self.relu(out)


class AlphaPredictor(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(AlphaPredictor, self).__init__()
        self.predictor = nn.Sequential(
            ResidualBlock(in_planes, out_planes, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_planes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.predictor(x)


if __name__ == '__main__':
    model = AlphaPredictor(3, 64)
    print(model)
    X = torch.randn(128, 3, 32, 32)
    y = model(X)
    print(y.shape)
