import torch.nn as nn
import torch


class Identity(nn.Module):
    def forward(self, x):
        return x


class Embedding(nn.Module):
    def __init__(self, in_channel, out_channel, linear_in, linear_out, num_classes):
        super(Embedding, self).__init__()
        self.linear_block = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channel),
        )

        self.linear_emb = nn.Linear(linear_in, linear_out, bias=False)
        self.bn1d = nn.BatchNorm1d(linear_out)

        self.weight = torch.nn.Parameter(
            torch.normal(0, 0.01, (num_classes, linear_out))
        )

    def forward(self, x):
        x = self.linear_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_emb(x)
        x = self.bn1d(x)

        return x, self.weight


class SimpleEmbedding(nn.Module):
    def __init__(self, in_channel, out_channel, linear_in, linear_out, num_classes):
        super(SimpleEmbedding, self).__init__()
        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
        )

        self.linear_emb = nn.Linear(linear_in, linear_out, bias=False)
        self.weight = torch.nn.Parameter(
            torch.normal(0, 0.01, (num_classes, linear_out))
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear_emb(x)

        return x, self.weight
