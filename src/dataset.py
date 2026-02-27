"""
Universal Dataset Loader for Image Classification
Supports multiple datasets with consistent preprocessing
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image, ImageFile
import numpy as np
from pathlib import Path

# Allow loading truncated images (some Food101 images may be incomplete)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageClassificationDataset(Dataset):
    """Generic image classification dataset loader.

    Expects directory structure:
    root/
        class1/
            img1.jpg
            img2.jpg
        class2/
            img1.jpg
            ...
    """

    def __init__(self, root_dir, transform=None, max_samples_per_class=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.class_names = []
        self.class_to_idx = {}

        # Discover classes
        classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_names = classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        # Collect samples
        for cls in classes:
            cls_dir = self.root_dir / cls
            images = [f for f in cls_dir.iterdir()
                      if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')]
            if max_samples_per_class:
                images = images[:max_samples_per_class]
            for img_path in images:
                self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def num_classes(self):
        return len(self.class_names)


def get_transforms(img_size=224, is_training=True):
    """Get standard transforms for training/evaluation."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def create_dataloaders(data_dir, img_size=224, batch_size=32,
                       train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                       num_workers=0, seed=42):
    """Create train/val/test dataloaders from a single directory."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Full dataset without transforms first (for splitting)
    full_dataset = ImageClassificationDataset(data_dir)
    total = len(full_dataset)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Apply transforms
    train_transform = get_transforms(img_size, is_training=True)
    eval_transform = get_transforms(img_size, is_training=False)

    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, eval_transform)
    test_dataset = TransformSubset(test_subset, eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    info = {
        'num_classes': full_dataset.num_classes,
        'class_names': full_dataset.class_names,
        'total_samples': total,
        'train_samples': n_train,
        'val_samples': n_val,
        'test_samples': n_test,
    }

    return train_loader, val_loader, test_loader, info


class TransformSubset(Dataset):
    """Apply transforms to a subset."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
