from glob import glob
from pathlib import Path

import albumentations as A
import numpy as np
import tifffile
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


def get_image_mask_pairs(data_dir: str):
    """Find MRI images and their corresponding tumor masks."""

    all_files = glob(
        str(Path(data_dir) / "**" / "*.tif"),
        recursive=True,
    )

    mask_files = [
        path
        for path in all_files
        if Path(path).stem.endswith("_mask")
    ]

    mask_lookup = {
        Path(path).stem.removesuffix("_mask"): path
        for path in mask_files
    }

    pairs = []

    for image_path in all_files:
        image_name = Path(image_path).stem

        if image_name.endswith("_mask"):
            continue

        if image_name in mask_lookup:
            pairs.append(
                (image_path, mask_lookup[image_name])
            )

    pairs.sort()

    print(f"Found image-mask pairs: {len(pairs)}")

    return pairs


class BrainTumorDataset(Dataset):

    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        image_path, mask_path = self.pairs[idx]

        image = tifffile.imread(image_path)
        mask = tifffile.imread(mask_path)

        # --------------------------------------------------
        # Image
        # --------------------------------------------------

        # Dataset images are already:
        # H x W x 3
        image = image.astype(np.uint8)

        # --------------------------------------------------
        # Mask
        # --------------------------------------------------

        # Convert mask to binary:
        # 0 = background
        # 1 = tumor
        mask = (mask > 0).astype(np.uint8)

        # --------------------------------------------------
        # Augmentation / preprocessing
        # --------------------------------------------------

        if self.transform:

            transformed = self.transform(
                image=image,
                mask=mask,
            )

            image = transformed["image"]
            mask = transformed["mask"]

        # --------------------------------------------------
        # Ensure tensor format
        # --------------------------------------------------

        if not isinstance(image, torch.Tensor):

            image = torch.from_numpy(
                image.transpose(2, 0, 1)
            ).float() / 255.0

        if not isinstance(mask, torch.Tensor):

            mask = torch.from_numpy(mask).float()

        # Mask:
        # H x W → 1 x H x W
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Ensure mask is float
        mask = mask.float()

        return image, mask


def get_train_transform(image_size=256):

    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
            ),

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.2),

            A.RandomRotate90(p=0.5),

            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                rotate=(-15, 15),
                p=0.5,
            ),

            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),

            ToTensorV2(),
        ]
    )


def get_val_transform(image_size=256):

    return A.Compose(
        [
            A.Resize(
                height=image_size,
                width=image_size,
            ),

            A.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),

            ToTensorV2(),
        ]
    )


def create_dataloaders(
    data_dir,
    batch_size=16,
    image_size=256,
    num_workers=0,
):

    pairs = get_image_mask_pairs(data_dir)

    if not pairs:
        raise RuntimeError(
            "No image-mask pairs found."
        )

    # Reproducible shuffle
    rng = np.random.default_rng(42)

    indices = rng.permutation(len(pairs))

    pairs = [pairs[i] for i in indices]

    # --------------------------------------------------
    # Split
    # --------------------------------------------------

    n = len(pairs)

    train_end = int(0.70 * n)
    val_end = int(0.85 * n)

    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    print(f"Train: {len(train_pairs)}")
    print(f"Val:   {len(val_pairs)}")
    print(f"Test:  {len(test_pairs)}")

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    train_dataset = BrainTumorDataset(
        train_pairs,
        transform=get_train_transform(image_size),
    )

    val_dataset = BrainTumorDataset(
        val_pairs,
        transform=get_val_transform(image_size),
    )

    test_dataset = BrainTumorDataset(
        test_pairs,
        transform=get_val_transform(image_size),
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader