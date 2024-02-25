import torch
from torch import nn


class Gated(nn.Module):
    def __init__(self, dims):
        super(Gated, self).__init__()
        self.layer1 = FC(dims=dims)
        self.layer2 = FC(dims=dims)
        self.sig = nn.Sigmoid()

    def forward(self, x, y):
        z1 = self.layer1(x)
        z2 = self.layer2(y)
        z = torch.sigmoid(torch.add(z1, z2))
        res = torch.add(torch.mul(x, 1 - z), torch.mul(y, z))
        return res

class FC(nn.Module):
    def __init__(self, dims, activation=None, dropout=0.):
        super(FC, self).__init__()
        self.hidden = dims * 3
        self.layers = nn.Sequential(
            nn.Linear(dims, self.hidden),
            nn.LayerNorm(self.hidden), nn.LeakyReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.LayerNorm(self.hidden), nn.LeakyReLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.LayerNorm(self.hidden), nn.LeakyReLU(),
            nn.Linear(self.hidden, dims),
            nn.LayerNorm(dims), nn.LeakyReLU()
        )#way2
        # self.layers = nn.Sequential(
        #     nn.Linear(dims, self.hidden),
        #     nn.LayerNorm(self.hidden), nn.ReLU(),
        #     nn.Linear(self.hidden, self.hidden),
        #     nn.LayerNorm(self.hidden), nn.ReLU(),
        #     nn.Linear(self.hidden, dims),
        #     nn.LayerNorm(dims), nn.ReLU()
        # )#way2
        self.bn = nn.BatchNorm2d(dims)
        self.act = activation if activation is not None else nn.LeakyReLU()
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else nn.Identity()

    def forward(self, x):
        x = self.layers(x.permute(0, 2, 3, 1))
        y = self.act(self.bn(x.permute(0, 3, 1, 2)))
        y = self.dropout(y)
        return y
