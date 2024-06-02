import torch.nn as nn


# N=(W-F+2P)/S+1
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 32, 3, 1, 1),  # size = 32*32*32
            nn.ReLU(),
            # Conv2
            nn.Conv2d(32, 64, 3, 1, 1),  # size = 32*32*64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # size = 16*16*64
            # Conv3
            nn.Conv2d(64, 128, 3, 1, 1),  # size = 16*16*128
            nn.ReLU(),
            # Conv4
            nn.Conv2d(128, 128, 3, 1, 1),  # size = 16*16*128
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # size = 8*8*128
            # Conv5
            nn.Conv2d(128, 256, 3, 1, 1),  # size = 8*8*256
            nn.ReLU(),
            # Conv6
            nn.Conv2d(256, 256, 3, 1, 1),  # size = 8*8*256
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # size = 4*4*256
        )
        self.classifier = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
