import torch
from tqdm import tqdm

from brain_seg.dataset import create_dataloaders
from brain_seg.model import UNet
from brain_seg.metrics import dice_score, iou_score


DATA_DIR = "data/raw"
CHECKPOINT = "outputs/checkpoints/best_model.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 8


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
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

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Checkpoint validation Dice: "
        f"{checkpoint['val_dice']:.4f}"
    )

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():

        for images, masks in tqdm(
            test_loader,
            desc="Testing",
        ):

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)

            dice = dice_score(
                logits,
                masks,
            )

            iou = iou_score(
                logits,
                masks,
            )

            total_dice += dice
            total_iou += iou

    num_batches = len(test_loader)

    test_dice = total_dice / num_batches
    test_iou = total_iou / num_batches

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print("\n" + "=" * 40)
    print("TEST RESULTS")
    print("=" * 40)

    print(
        f"Dice Score : {test_dice:.4f}"
    )

    print(
        f"IoU Score  : {test_iou:.4f}"
    )

    print("=" * 40)


if __name__ == "__main__":
    main()