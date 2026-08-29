from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from brain_seg.dataset import create_dataloaders
from brain_seg.model import UNet


DATA_DIR = "data/raw"
CHECKPOINT = "outputs/checkpoints/best_model.pth"
OUTPUT_DIR = Path("outputs/predictions")

IMAGE_SIZE = 256
BATCH_SIZE = 1
NUM_SAMPLES = 8
SEED = 42


def main():
    THRESHOLD = 0.60

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load test dataset
    # --------------------------------------------------

    _, _, test_loader = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=0,
    )

    # --------------------------------------------------
    # Load trained model
    # --------------------------------------------------

    model = UNet(
        in_channels=3,
        out_channels=1,
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
    # Random sample indices
    # --------------------------------------------------

    rng = np.random.default_rng(SEED)

    total_images = len(test_loader.dataset)

    random_indices = rng.choice(
        total_images,
        size=min(NUM_SAMPLES, total_images),
        replace=False,
    )

    random_indices = sorted(random_indices)

    print(
        f"Random test indices: {random_indices}"
    )

    # --------------------------------------------------
    # Generate predictions
    # --------------------------------------------------

    for sample_number, index in enumerate(
        random_indices,
        start=1,
    ):

        image, mask = test_loader.dataset[index]

        image_input = image.unsqueeze(0).to(device)

        with torch.no_grad():

            logits = model(image_input)

            probability = torch.sigmoid(logits)

            prediction = (
                probability > THRESHOLD
            ).float()

        # --------------------------------------------------
        # Convert tensors for visualization
        # --------------------------------------------------

        image_np = image.cpu()

        image_np = image_np.permute(
            1, 2, 0
        ).numpy()

        mask_np = mask.cpu().numpy().squeeze()

        prediction_np = (
            prediction.cpu()
            .numpy()
            .squeeze()
        )

        # Normalize image
        image_np = (
            image_np - image_np.min()
        ) / (
            image_np.max()
            - image_np.min()
            + 1e-8
        )

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(12, 4),
        )

        axes[0].imshow(image_np)
        axes[0].set_title("MRI")

        axes[1].imshow(
            mask_np,
            cmap="gray",
        )
        axes[1].set_title(
            "Ground Truth"
        )

        axes[2].imshow(
            prediction_np,
            cmap="gray",
        )
        axes[2].set_title(
            "Prediction"
        )

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()

        output_path = (
            OUTPUT_DIR
            / f"random_prediction_{sample_number}.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":
    main()