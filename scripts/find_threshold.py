import torch
from tqdm import tqdm

from brain_seg.dataset import create_dataloaders
from brain_seg.model import UNet


DATA_DIR = "data/raw"
CHECKPOINT = "outputs/checkpoints/best_model.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 8

THRESHOLDS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]


def calculate_dice(predictions, targets, smooth=1e-6):

    predictions = predictions.flatten(1)
    targets = targets.flatten(1)

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2 * intersection + smooth
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return dice.mean().item()


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # --------------------------------------------------
    # Validation data
    # --------------------------------------------------

    _, val_loader, _ = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=0,
    )

    # --------------------------------------------------
    # Model
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

    # Store predictions once
    all_probabilities = []
    all_masks = []

    with torch.no_grad():

        for images, masks in tqdm(
            val_loader,
            desc="Running validation",
        ):

            images = images.to(device)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            all_probabilities.append(
                probabilities.cpu()
            )

            all_masks.append(
                masks.cpu()
            )

    probabilities = torch.cat(
        all_probabilities
    )

    masks = torch.cat(
        all_masks
    )

    # --------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------

    best_threshold = 0.5
    best_dice = -1

    print("\nThreshold results")
    print("-" * 30)

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).float()

        dice = calculate_dice(
            predictions,
            masks,
        )

        print(
            f"Threshold {threshold:.2f} "
            f"→ Dice {dice:.4f}"
        )

        if dice > best_dice:

            best_dice = dice
            best_threshold = threshold

    print("\n" + "=" * 40)

    print(
        f"Best threshold: "
        f"{best_threshold:.2f}"
    )

    print(
        f"Best validation Dice: "
        f"{best_dice:.4f}"
    )

    print("=" * 40)


if __name__ == "__main__":
    main()