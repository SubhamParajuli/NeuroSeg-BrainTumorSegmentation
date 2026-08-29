from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from brain_seg.dataset import create_dataloaders
from brain_seg.model import UNet


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = "data/raw"
CHECKPOINT = Path(
    "outputs/checkpoints/best_model.pth"
)

OUTPUT_DIR = Path(
    "outputs/evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

IMAGE_SIZE = 256
BATCH_SIZE = 8
THRESHOLD = 0.60
NUM_SAMPLES = 6


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")


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


# --------------------------------------------------
# Collect random samples
# --------------------------------------------------

images, masks = next(
    iter(test_loader)
)

indices = torch.randperm(
    images.size(0)
)[:NUM_SAMPLES]


images = images[indices]
masks = masks[indices]


# --------------------------------------------------
# Prediction
# --------------------------------------------------

with torch.inference_mode():

    logits = model(
        images.to(device)
    )

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= THRESHOLD
    ).float()


# --------------------------------------------------
# Plot
# --------------------------------------------------

fig, axes = plt.subplots(
    NUM_SAMPLES,
    4,
    figsize=(14, 3 * NUM_SAMPLES),
)


for i in range(NUM_SAMPLES):

    # ----------------------------------------------
    # Image
    # ----------------------------------------------

    image = (
        images[i]
        .permute(1, 2, 0)
        .numpy()
    )

    # Undo normalization
    image = (
        image * 0.5 + 0.5
    )

    image = np.clip(
        image,
        0,
        1,
    )

    # ----------------------------------------------
    # Ground truth
    # ----------------------------------------------

    ground_truth = (
        masks[i]
        .squeeze()
        .numpy()
    )

    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    prediction = (
        predictions[i]
        .squeeze()
        .cpu()
        .numpy()
    )

    # ----------------------------------------------
    # Overlay
    # ----------------------------------------------

    overlay = image.copy()

    overlay[
        prediction == 1
    ] = (
        overlay[
            prediction == 1
        ] * 0.4
        + np.array([1.0, 0.0, 0.0])
        * 0.6
    )

    # ----------------------------------------------
    # Draw
    # ----------------------------------------------

    axes[i, 0].imshow(image)
    axes[i, 0].set_title("MRI")

    axes[i, 1].imshow(
        ground_truth,
        cmap="gray",
    )
    axes[i, 1].set_title("Ground Truth")

    axes[i, 2].imshow(
        prediction,
        cmap="gray",
    )
    axes[i, 2].set_title("Prediction")

    axes[i, 3].imshow(overlay)
    axes[i, 3].set_title("Prediction Overlay")

    for j in range(4):
        axes[i, j].axis("off")


plt.suptitle(
    "NeuroSeg — Random Test Set Predictions",
    fontsize=16,
)

plt.tight_layout()

output_file = (
    OUTPUT_DIR
    / "prediction_samples.png"
)

plt.savefig(
    output_file,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(
    f"Saved: {output_file}"
)