import torch
from torch import nn
import torchvision.models as models
from config import Common, Train

# 引入rest50模型
# 直接从官网下载，默认到 ~/.cache/torch/hub/checkpoints/ 目录下 的 resnet50-11ad3fa6.pth
net = models.resnet50(weights='DEFAULT')


class WeatherModel(nn.Module):
    def __init__(self, net):
        super(WeatherModel, self).__init__()
        # resnet50
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(1000, 8)
        self.output = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.net(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.output(x)
        return x


model = WeatherModel(net)
