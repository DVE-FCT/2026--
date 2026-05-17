# 自定义数据加载器
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import Common
from config import Train
import os
from PIL import Image
import numpy as np

# ============================================================
# 验证/测试集 transform（无数据增强）
# ============================================================
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ============================================================
# 训练集 transform（条件数据增强）
# ============================================================
if Train.data_augmentation_enabled:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.25,
            contrast=0.25,
            saturation=0.2,
            hue=0.03
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
else:
    train_transform = test_transform


class WeatherDataSet(Dataset):
    '''
    自定义DataSet（惰性加载，按需读取图片，每类最多 max_per_class 张）
    '''

    def __init__(self, max_per_class=1000, transform=None):
        self.samples = []  # (image_path, label_index)
        self.transform = transform
        for d in os.listdir(Common.basePath):
            dir_path = os.path.join(Common.basePath, d)
            if not os.path.isdir(dir_path):
                continue
            categoryIndex = Common.labels.index(d)
            image_files = [f for f in os.listdir(dir_path)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]
            if len(image_files) > max_per_class:
                image_files = np.random.choice(image_files, max_per_class, replace=False).tolist()
            for imagePath in image_files:
                self.samples.append((os.path.join(dir_path, imagePath), categoryIndex))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        imagePath, categoryIndex = self.samples[idx]
        image = Image.open(imagePath).convert('RGB')
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = test_transform(image)
        image.close()
        label = [0] * 8
        label[categoryIndex] = 1
        label_tensor = torch.tensor(label, dtype=torch.float)
        return image_tensor, label_tensor


# ============================================================
# 构建数据集（每类最多 1000 张）
# ============================================================
# 训练集：有数据增强
train_dataset = WeatherDataSet(max_per_class=1000, transform=train_transform)
# 验证/测试集：无增强
val_test_dataset = WeatherDataSet(max_per_class=1000, transform=test_transform)

# 分层划分（保证每类 70/15/15 比例）
train_indices, val_indices, test_indices = [], [], []
if Train.stratified_split_enabled:
    for label_idx in range(len(Common.labels)):
        label_indices = [i for i, (_, l) in enumerate(train_dataset.samples) if l == label_idx]
        np.random.shuffle(label_indices)
        n = len(label_indices)
        train_n = int(n * 0.7)
        val_n = int(n * 0.15)
        train_indices.extend(label_indices[:train_n])
        val_indices.extend(label_indices[train_n:train_n + val_n])
        test_indices.extend(label_indices[train_n + val_n:])
else:
    all_indices = list(range(len(train_dataset.samples)))
    np.random.shuffle(all_indices)
    n = len(all_indices)
    train_n = int(n * 0.7)
    val_n = int(n * 0.15)
    train_indices = all_indices[:train_n]
    val_indices = all_indices[train_n:train_n + val_n]
    test_indices = all_indices[train_n + val_n:]

train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(val_test_dataset, val_indices)
test_subset = torch.utils.data.Subset(val_test_dataset, test_indices)

# ============================================================
# 数据加载器
# ============================================================
trainLoader = DataLoader(
    train_subset,
    batch_size=Train.batch_size,
    shuffle=True,
    num_workers=Train.num_workers,
    pin_memory=True
)

valLoader = DataLoader(
    val_subset,
    batch_size=Train.batch_size,
    shuffle=False,
    num_workers=Train.num_workers,
    pin_memory=True
)

testLoader = DataLoader(
    test_subset,
    batch_size=Train.batch_size,
    shuffle=False,
    num_workers=Train.num_workers,
    pin_memory=True
)
