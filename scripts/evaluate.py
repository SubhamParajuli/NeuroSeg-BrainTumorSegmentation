from pathlib import Path

import json
import torch
from tqdm import tqdm

from brain_seg.dataset import create_dataloaders
from brain_seg.metrics import segmentation_metrics
from brain_seg.model import UNet


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = "data/raw"
CHECKPOINT = Path(
    "outputs/checkpoints/best_model.pth"
)

IMAGE_SIZE = 256
BATCH_SIZE = 8
THRESHOLD = 0.60

OUTPUT_DIR = Path(
    "outputs/evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# --------------------------------------------------
# Dataset
# --------------------------------------------------

_, _, test_loader = create_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = UNet(
    in_channels=3,
    out_channels=1,
    pretrained=False,
).to(device)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    f"Loaded checkpoint from epoch "
    f"{checkpoint.get('epoch', 'unknown')}"
)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

metric_totals = {
    "dice": 0.0,
    "iou": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "specificity": 0.0,
    "accuracy": 0.0,
}

num_batches = 0


with torch.inference_mode():

    for images, masks in tqdm(
        test_loader,
        desc="Testing",
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        logits = model(images)

        metrics = segmentation_metrics(
            logits,
            masks,
            threshold=THRESHOLD,
        )

        for key in metric_totals:
            metric_totals[key] += metrics[key]

        num_batches += 1


# --------------------------------------------------
# Average metrics
# --------------------------------------------------

final_metrics = {
    key: value / num_batches
    for key, value in metric_totals.items()
}


# --------------------------------------------------
# Add experiment information
# --------------------------------------------------

final_metrics.update(
    {
        "model": "ResNet18-UNet",
        "image_size": IMAGE_SIZE,
        "threshold": THRESHOLD,
        "test_samples": len(
            test_loader.dataset
        ),
        "parameters": sum(
            p.numel()
            for p in model.parameters()
        ),
        "device": str(device),
    }
)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print("=" * 40)
print("NEUROSEG EVALUATION")
print("=" * 40)

print(
    f"Model        : ResNet18-UNet"
)

print(
    f"Device       : {device}"
)

print(
    f"Threshold    : {THRESHOLD}"
)

print(
    f"Test samples : "
    f"{len(test_loader.dataset)}"
)

print()

print(
    f"Dice         : "
    f"{final_metrics['dice']:.4f}"
)

print(
    f"IoU          : "
    f"{final_metrics['iou']:.4f}"
)

print(
    f"Precision    : "
    f"{final_metrics['precision']:.4f}"
)

print(
    f"Recall       : "
    f"{final_metrics['recall']:.4f}"
)

print(
    f"Specificity  : "
    f"{final_metrics['specificity']:.4f}"
)

print(
    f"Accuracy     : "
    f"{final_metrics['accuracy']:.4f}"
)

print("=" * 40)


# --------------------------------------------------
# Save JSON
# --------------------------------------------------

metrics_path = (
    OUTPUT_DIR / "metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        final_metrics,
        file,
        indent=4,
    )


print(
    f"\nSaved metrics: {metrics_path}"
)