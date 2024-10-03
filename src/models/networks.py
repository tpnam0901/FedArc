import torch.nn as nn
import torchvision.models as models
from configs.base import Config
from .modules import Identity, Embedding, SimpleEmbedding


class ResNet18(nn.Module):
    def __init__(self, cfg: Config):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.resnet.fc = Identity()
        self.embedding = Embedding(
            cfg.embedding_channel_in,
            cfg.embedding_channel_out,
            cfg.embedding_linear_in,
            cfg.embedding_linear_out,
            cfg.num_classes,
        )
        self.classifier = nn.Linear(cfg.embedding_linear_out, cfg.num_classes)

    def forward(self, x):
        x = self.resnet(x)
        feat, feat_weight = self.embedding(x.view(x.size(0), 512, 1, 1))
        logits = self.classifier(feat)
        return logits, feat, feat_weight

class SimpleCNN_MNIST(nn.Module):
    def __init__(self, cfg: Config):
        super(SimpleCNN_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.embedding = SimpleEmbedding(
            cfg.embedding_channel_in,
            cfg.embedding_channel_out,
            cfg.embedding_linear_in,
            cfg.embedding_linear_out,
            cfg.num_classes,
        )
        self.classifier = nn.Linear(cfg.embedding_linear_out, cfg.num_classes)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        feat, feat_weight = self.embedding(x)
        logits = self.classifier(feat)
        return logits, feat, feat_weight

