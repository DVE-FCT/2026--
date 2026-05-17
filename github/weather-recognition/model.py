import torch
from torch import nn
import torchvision.models as models
from config import Common, Train

net = models.resnet50(weights='DEFAULT')
net.fc = nn.Identity()  # 让 net(x) 输出 2048 维特征（而非 1000 维分类输出）


class WeatherModel(nn.Module):
    def __init__(self, net):
        super(WeatherModel, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(2048, 8)  # 2048 维特征 -> 8 类

    def forward(self, x):
        x = self.net(x)  # 输出 2048 维特征
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)  # 输出原始 logits（无 Softmax，CrossEntropyLoss/FocalLoss 内部处理）
        return x


model = WeatherModel(net)
