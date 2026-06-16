import os
from typing import Dict, Iterator, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import Common, SEED, Train


# ============================================================
# Transforms
# ============================================================
_IMG_SIZE = Train.get_image_size()
_RESIZE = int(round(_IMG_SIZE / 0.875))

test_transform = transforms.Compose(
    [
        transforms.Resize(_RESIZE),
        transforms.CenterCrop(_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

if Train.data_augmentation_enabled:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                _IMG_SIZE,
                scale=(0.7, 1.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.2,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
else:
    train_transform = test_transform


class WeatherDataSet(Dataset):
    """Lazy image dataset. Samples are stored as (path, class_index)."""

    def __init__(self, max_per_class=None, transform=None):
        self.samples = []
        self.transform = transform

        for class_name in sorted(os.listdir(Common.basePath)):
            class_dir = os.path.join(Common.basePath, class_name)
            if not os.path.isdir(class_dir):
                continue
            if class_name not in Common.labels:
                continue

            class_index = Common.labels.index(class_name)
            image_files = sorted(
                f
                for f in os.listdir(class_dir)
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                )
            )

            if max_per_class and len(image_files) > max_per_class:
                rng = np.random.default_rng(SEED + class_index)
                image_files = rng.choice(
                    image_files,
                    max_per_class,
                    replace=False,
                ).tolist()

            for image_name in image_files:
                self.samples.append(
                    (os.path.join(class_dir, image_name), class_index)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, class_index = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = (
                self.transform(image)
                if self.transform is not None
                else test_transform(image)
            )

        # Keep one-hot output for compatibility with existing test scripts.
        label = torch.zeros(len(Common.labels), dtype=torch.float32)
        label[class_index] = 1.0
        return image_tensor, label

    @classmethod
    def from_indices(cls, base_dataset, indices, transform=None):
        instance = cls.__new__(cls)
        instance.samples = [
            base_dataset.samples[i]
            for i in indices
        ]
        instance.transform = transform
        return instance


# ============================================================
# Deterministic split
# ============================================================
base_dataset = WeatherDataSet(transform=None)
rng = np.random.default_rng(SEED)

train_indices: List[int] = []
val_indices: List[int] = []
test_indices: List[int] = []

if Train.stratified_split_enabled:
    for class_index in range(len(Common.labels)):
        class_indices = [
            i
            for i, (_, label) in enumerate(base_dataset.samples)
            if label == class_index
        ]
        class_indices = np.asarray(class_indices, dtype=np.int64)
        rng.shuffle(class_indices)

        n = len(class_indices)
        train_n = int(n * 0.70)
        val_n = int(n * 0.15)

        train_indices.extend(class_indices[:train_n].tolist())
        val_indices.extend(
            class_indices[train_n : train_n + val_n].tolist()
        )
        test_indices.extend(
            class_indices[train_n + val_n :].tolist()
        )
else:
    all_indices = np.arange(len(base_dataset), dtype=np.int64)
    rng.shuffle(all_indices)

    n = len(all_indices)
    train_n = int(n * 0.70)
    val_n = int(n * 0.15)

    train_indices = all_indices[:train_n].tolist()
    val_indices = all_indices[
        train_n : train_n + val_n
    ].tolist()
    test_indices = all_indices[
        train_n + val_n :
    ].tolist()

# Shuffle each split deterministically after class-wise concatenation.
rng.shuffle(train_indices)
rng.shuffle(val_indices)
rng.shuffle(test_indices)

train_dataset = WeatherDataSet.from_indices(
    base_dataset,
    train_indices,
    transform=train_transform,
)
val_dataset = WeatherDataSet.from_indices(
    base_dataset,
    val_indices,
    transform=test_transform,
)
test_dataset = WeatherDataSet.from_indices(
    base_dataset,
    test_indices,
    transform=test_transform,
)

# Same training images, but deterministic evaluation transform.
# Dynamic alpha uses this loader instead of randomly augmented trainLoader.
train_eval_dataset = WeatherDataSet.from_indices(
    base_dataset,
    train_indices,
    transform=test_transform,
)


# ============================================================
# Loaders
# ============================================================
_use_pin = Common.device.type == "cuda"
_num_workers = Train.num_workers
_num_classes = len(Common.labels)


class BalancedBatchSampler(torch.utils.data.Sampler):
    """Balanced batch sampler used only when SupCon is enabled."""

    def __init__(
        self,
        class_bins: Dict[int, List[int]],
        per_class: int,
    ):
        self.class_bins = class_bins
        self.per_class = per_class
        self.batch_size = per_class * _num_classes
        self.n_batches = len(train_dataset) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        local_rng = np.random.default_rng()

        batches = []
        for _ in range(self.n_batches):
            batch = []
            for class_index in range(_num_classes):
                pool = self.class_bins[class_index]
                replace = len(pool) < self.per_class
                chosen = local_rng.choice(
                    pool,
                    self.per_class,
                    replace=replace,
                )
                batch.extend(int(i) for i in chosen)
            local_rng.shuffle(batch)
            batches.append(batch)

        local_rng.shuffle(batches)
        yield from batches

    def __len__(self):
        return self.n_batches


common_loader_kwargs = dict(
    num_workers=_num_workers,
    pin_memory=_use_pin,
    persistent_workers=_num_workers > 0,
)

if _num_workers > 0:
    common_loader_kwargs["prefetch_factor"] = 2

if getattr(Train, "supcon_enabled", False):
    per_class = max(2, Train.batch_size // _num_classes)
    labels = [sample[1] for sample in train_dataset.samples]
    class_bins = {c: [] for c in range(_num_classes)}
    for index, label in enumerate(labels):
        class_bins[label].append(index)

    sampler = BalancedBatchSampler(class_bins, per_class)
    trainLoader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        **common_loader_kwargs,
    )
else:
    trainLoader = DataLoader(
        train_dataset,
        batch_size=Train.batch_size,
        shuffle=True,
        **common_loader_kwargs,
    )

trainEvalLoader = DataLoader(
    train_eval_dataset,
    batch_size=Train.batch_size,
    shuffle=False,
    **common_loader_kwargs,
)

valLoader = DataLoader(
    val_dataset,
    batch_size=Train.batch_size,
    shuffle=False,
    **common_loader_kwargs,
)

testLoader = DataLoader(
    test_dataset,
    batch_size=Train.batch_size,
    shuffle=False,
    **common_loader_kwargs,
)
