"""
Download and prepare all 4 datasets for ECA-MGNet experiments.
Uses torchvision dataset loaders to download and organize images.

The script downloads each dataset, selects 10 classes with the most images
(up to 300 images per class), and organizes them into the directory structure
expected by the training scripts:

    <output_dir>/
        flowers102/  class_000/ ... class_009/
        dtd/         banded/   ... zigzagged/
        food101/     apple_pie/ ... waffles/
        eurosat/     AnnualCrop/ ... SeaLake/

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --output_dir /path/to/datasets
"""
import os
import sys
import shutil
import zipfile
import tarfile
import argparse
import urllib.request
from pathlib import Path
import ssl

# Handle SSL verification issues on some platforms
ssl._create_default_https_context = ssl._create_unverified_context


def download_file(url, dest_path, desc=""):
    """Download a file with progress."""
    print(f"  Downloading {desc or url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  Done: {dest_path}")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract a zip file."""
    print(f"  Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"  Extracted to {extract_to}")


def extract_tar(tar_path, extract_to):
    """Extract a tar/tar.gz file."""
    print(f"  Extracting {tar_path}...")
    with tarfile.open(tar_path, 'r:*') as t:
        t.extractall(extract_to)
    print(f"  Extracted to {extract_to}")


# ===================================================================
# Dataset 1: Flowers102 (Oxford Flowers - via torchvision)
# ===================================================================
def prepare_flowers102(base_dir):
    """Prepare Oxford Flowers102 dataset (subset of ~3000 images)."""
    base_dir = Path(base_dir)
    dest = base_dir / "flowers102"
    if dest.exists() and any(dest.iterdir()):
        print("Flowers102 already exists, skipping...")
        return True

    print("\n[Dataset 1] Preparing Flowers102...")
    try:
        from torchvision.datasets import Flowers102
        from collections import Counter

        # Download via torchvision
        raw_dir = base_dir / "flowers102_raw"
        raw_dir.mkdir(exist_ok=True)

        train_ds = Flowers102(root=str(raw_dir), split='train', download=True)
        val_ds = Flowers102(root=str(raw_dir), split='val', download=True)
        test_ds = Flowers102(root=str(raw_dir), split='test', download=True)

        # Collect all items, pick top 10 classes by image count
        all_labels = []
        all_items = []
        for ds in [train_ds, val_ds, test_ds]:
            for idx in range(len(ds)):
                img, label = ds[idx]
                all_labels.append(label)
                all_items.append((img, label))

        counter = Counter(all_labels)
        top_classes = [cls for cls, _ in counter.most_common(10)]

        dest.mkdir(exist_ok=True)
        counts = {}
        for img, label in all_items:
            if label in top_classes:
                class_dir = dest / f"class_{label:03d}"
                class_dir.mkdir(exist_ok=True)
                count = counts.get(label, 0)
                img.save(class_dir / f"img_{count:04d}.jpg")
                counts[label] = count + 1

        total = sum(counts.values())
        print(f"  Flowers102 prepared: {total} images, {len(top_classes)} classes")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


# ===================================================================
# Dataset 2: DTD (Describable Textures Dataset - via torchvision)
# ===================================================================
def prepare_dtd(base_dir):
    """Prepare DTD texture dataset (subset of ~2800 images)."""
    base_dir = Path(base_dir)
    dest = base_dir / "dtd"
    if dest.exists() and any(dest.iterdir()):
        print("DTD already exists, skipping...")
        return True

    print("\n[Dataset 2] Preparing DTD (Textures)...")
    try:
        from torchvision.datasets import DTD
        from collections import Counter

        raw_dir = base_dir / "dtd_raw"
        raw_dir.mkdir(exist_ok=True)

        train_ds = DTD(root=str(raw_dir), split='train', download=True)
        val_ds = DTD(root=str(raw_dir), split='val', download=True)
        test_ds = DTD(root=str(raw_dir), split='test', download=True)

        # Organize into class directories (select top 10 classes)
        all_labels = []
        all_items = []
        for ds in [train_ds, val_ds, test_ds]:
            for idx in range(len(ds)):
                img_path = ds._image_files[idx]
                label = ds._labels[idx]
                all_labels.append(label)
                all_items.append((img_path, label))

        counter = Counter(all_labels)
        top_classes = [cls for cls, _ in counter.most_common(10)]
        class_names = train_ds.classes

        dest.mkdir(exist_ok=True)
        counts = {}
        for img_path, label in all_items:
            if label in top_classes:
                class_name = class_names[label]
                class_dir = dest / class_name
                class_dir.mkdir(exist_ok=True)
                count = counts.get(label, 0)
                shutil.copy2(str(img_path), class_dir / f"img_{count:04d}.jpg")
                counts[label] = count + 1

        total = sum(counts.values())
        print(f"  DTD prepared: {total} images, {len(top_classes)} classes")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


# ===================================================================
# Dataset 3: Food101 (subset - via torchvision)
# ===================================================================
def prepare_food101(base_dir):
    """Prepare Food101 subset (~3000 images, 10 classes)."""
    base_dir = Path(base_dir)
    dest = base_dir / "food101"
    if dest.exists() and any(dest.iterdir()):
        print("Food101 already exists, skipping...")
        return True

    print("\n[Dataset 3] Preparing Food101 (subset)...")
    try:
        from torchvision.datasets import Food101
        from collections import defaultdict

        raw_dir = base_dir / "food101_raw"
        raw_dir.mkdir(exist_ok=True)

        train_ds = Food101(root=str(raw_dir), split='train', download=True)
        test_ds = Food101(root=str(raw_dir), split='test', download=True)

        # Select 10 classes, up to 300 images each
        class_items = defaultdict(list)
        for ds in [train_ds, test_ds]:
            for idx in range(len(ds)):
                img_path = ds._image_files[idx]
                label = ds._labels[idx]
                class_items[label].append(img_path)

        sorted_classes = sorted(class_items.keys(),
                                key=lambda k: len(class_items[k]), reverse=True)
        selected = sorted_classes[:10]

        dest.mkdir(exist_ok=True)
        total = 0
        class_names = train_ds.classes

        for cls in selected:
            class_name = class_names[cls]
            class_dir = dest / class_name
            class_dir.mkdir(exist_ok=True)
            images = class_items[cls][:300]  # 300 per class
            for i, img_path in enumerate(images):
                shutil.copy2(str(img_path), class_dir / f"img_{i:04d}.jpg")
            total += len(images)

        print(f"  Food101 subset prepared: {total} images, 10 classes")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


# ===================================================================
# Dataset 4: EuroSAT (Satellite imagery - via torchvision)
# ===================================================================
def prepare_eurosat(base_dir):
    """Prepare EuroSAT dataset (~3000 images, 10 classes)."""
    base_dir = Path(base_dir)
    dest = base_dir / "eurosat"
    if dest.exists() and any(dest.iterdir()):
        print("EuroSAT already exists, skipping...")
        return True

    print("\n[Dataset 4] Preparing EuroSAT...")
    try:
        from torchvision.datasets import EuroSAT
        from collections import defaultdict

        raw_dir = base_dir / "eurosat_raw"
        raw_dir.mkdir(exist_ok=True)

        ds = EuroSAT(root=str(raw_dir), download=True)

        class_items = defaultdict(list)
        for idx in range(len(ds)):
            img_path = ds._image_files[idx]
            label = ds._labels[idx]
            class_items[label].append(img_path)

        dest.mkdir(exist_ok=True)
        total = 0
        class_names = ds.classes

        for cls in range(len(class_names)):
            class_name = class_names[cls]
            class_dir = dest / class_name
            class_dir.mkdir(exist_ok=True)
            images = class_items[cls][:300]  # 300 per class
            for i, img_path in enumerate(images):
                shutil.copy2(str(img_path), class_dir / f"img_{i:04d}.jpg")
            total += len(images)

        print(f"  EuroSAT prepared: {total} images, {len(class_names)} classes")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main(output_dir):
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading and preparing datasets for ECA-MGNet")
    print(f"Output directory: {base_dir.resolve()}")
    print("=" * 60)

    status = {}
    status['flowers102'] = prepare_flowers102(base_dir)
    status['dtd'] = prepare_dtd(base_dir)
    status['food101'] = prepare_food101(base_dir)
    status['eurosat'] = prepare_eurosat(base_dir)

    print("\n" + "=" * 60)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 60)
    for name, success in status.items():
        label = "OK" if success else "FAILED"
        if success:
            path = base_dir / name
            if path.exists():
                classes = [d for d in path.iterdir() if d.is_dir()]
                n_images = sum(len(list(c.iterdir())) for c in classes)
                print(f"  {name:<12}: {label} - {n_images} images, {len(classes)} classes")
            else:
                print(f"  {name:<12}: {label}")
        else:
            print(f"  {name:<12}: {label}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download and prepare all 4 datasets for ECA-MGNet experiments.'
    )
    parser.add_argument('--output_dir', type=str, default='datasets',
                        help='Directory where datasets will be saved. Default: datasets/')
    args = parser.parse_args()
    main(args.output_dir)
