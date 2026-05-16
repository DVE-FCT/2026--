# 自定义数据加载器
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import Common
from config import Train
import os
from PIL import Image
import torch.utils.data as Data
import numpy
import random

# 定义数据处理transform（ImageNet 标准预处理）
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class WeatherDataSet(Dataset):
    '''
    自定义DataSet（惰性加载，按需读取图片，每类最多1000张）
    '''

    def __init__(self, max_per_class=1000):
        self.samples = []  # (image_path, label_index)
        for d in os.listdir(Common.basePath):
            dir_path = Common.basePath + d + "/"
            if not os.path.isdir(dir_path):
                continue
            categoryIndex = Common.labels.index(d)
            image_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            if len(image_files) > max_per_class:
                image_files = random.sample(image_files, max_per_class)
            for imagePath in image_files:
                self.samples.append((dir_path + imagePath, categoryIndex))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imagePath, categoryIndex = self.samples[idx]
        image = Image.open(imagePath).convert('RGB')
        image_tensor = transform(image)
        image.close()
        label = [0] * 8
        label[categoryIndex] = 1
        label_tensor = torch.tensor(label, dtype=torch.float)
        return image_tensor, label_tensor


def splitData(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    '''
    分割数据集为训练集、验证集、测试集
    '''
    total_length = len(dataset)
    train_length = int(total_length * train_ratio)
    val_length = int(total_length * val_ratio)
    test_length = total_length - train_length - val_length

    train_dataset, val_dataset, test_dataset = Data.random_split(
        dataset=dataset,
        lengths=[train_length, val_length, test_length]
    )
    return train_dataset, val_dataset, test_dataset


# 1. 构建数据集（每类最多1000张）
dataset = WeatherDataSet(max_per_class=1000)
# 2. 分割数据集
train_dataset, val_dataset, test_dataset = splitData(dataset)
# 3. 训练数据集加载器（pin_memory 加速 CPU -> GPU 传输）
trainLoader = DataLoader(train_dataset, batch_size=Train.batch_size, shuffle=True, num_workers=Train.num_workers, pin_memory=True)
# 4. 验证集数据加载器
valLoader = DataLoader(val_dataset, batch_size=Train.batch_size, shuffle=False, num_workers=Train.num_workers, pin_memory=True)
# 5. 测试集数据加载器
testLoader = DataLoader(test_dataset, batch_size=Train.batch_size, shuffle=False, num_workers=Train.num_workers, pin_memory=True)
