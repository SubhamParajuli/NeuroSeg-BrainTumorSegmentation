import random
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from brain_seg.dataset import create_dataloaders
from brain_seg.losses import BCEDiceLoss
from brain_seg.metrics import dice_score, iou_score
from brain_seg.model import UNet


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data/raw"
CHECKPOINT_DIR = Path("outputs/checkpoints")
PLOT_DIR = Path("outputs/plots")

IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_EPOCHS = 1      #number of epochs

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

SEED = 42

MLFLOW_EXPERIMENT = "brain-tumor-segmentation"


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

        print(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    else:

        device = torch.device("cpu")

        print("WARNING: CUDA unavailable. Using CPU.")

    return device


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    scaler,
    device,
):

    model.train()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for images, masks in progress:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):

            logits = model(images)

            loss = loss_fn(
                logits,
                masks,
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        batch_dice = dice_score(
            logits.detach(),
            masks,
        )

        batch_iou = iou_score(
            logits.detach(),
            masks,
        )

        running_loss += loss.item()

        running_dice += batch_dice

        running_iou += batch_iou

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            dice=f"{batch_dice:.4f}",
        )

    num_batches = len(loader)

    return (
        running_loss / num_batches,
        running_dice / num_batches,
        running_iou / num_batches,
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    loss_fn,
    device,
):

    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0

    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for images, masks in progress:

        images = images.to(
            device,
            non_blocking=True,
        )

        masks = masks.to(
            device,
            non_blocking=True,
        )

        with autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):

            logits = model(images)

            loss = loss_fn(
                logits,
                masks,
            )

        batch_dice = dice_score(
            logits,
            masks,
        )

        batch_iou = iou_score(
            logits,
            masks,
        )

        running_loss += loss.item()

        running_dice += batch_dice

        running_iou += batch_iou

    num_batches = len(loader)

    return (
        running_loss / num_batches,
        running_dice / num_batches,
        running_iou / num_batches,
    )


# ============================================================
# Main
# ============================================================

def main():

    set_seed(SEED)

    device = get_device()

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print("\nLoading dataset...")

    train_loader, val_loader, _ = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nCreating U-Net...")

    model = UNet(
        in_channels=3,
        out_channels=1,
    ).to(device)

    print(
        f"Parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_fn = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # --------------------------------------------------------
    # Mixed precision
    # --------------------------------------------------------

    scaler = GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    with mlflow.start_run():

        # Log parameters

        mlflow.log_params(
            {
                "model": "U-Net",
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": NUM_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "optimizer": "AdamW",
                "loss": "BCE + Dice",
                "seed": SEED,
            }
        )

        best_val_dice = -1.0

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_dice": [],
            "val_dice": [],
            "train_iou": [],
            "val_iou": [],
        }

        # ----------------------------------------------------
        # Training loop
        # ----------------------------------------------------

        for epoch in range(1, NUM_EPOCHS + 1):

            print(
                f"\nEpoch "
                f"{epoch}/{NUM_EPOCHS}"
            )

            train_loss, train_dice, train_iou = (
                train_one_epoch(
                    model,
                    train_loader,
                    loss_fn,
                    optimizer,
                    scaler,
                    device,
                )
            )

            val_loss, val_dice, val_iou = validate(
                model,
                val_loader,
                loss_fn,
                device,
            )

            scheduler.step(val_dice)

            current_lr = optimizer.param_groups[0]["lr"]

            # Store history

            history["train_loss"].append(
                train_loss
            )

            history["val_loss"].append(
                val_loss
            )

            history["train_dice"].append(
                train_dice
            )

            history["val_dice"].append(
                val_dice
            )

            history["train_iou"].append(
                train_iou
            )

            history["val_iou"].append(
                val_iou
            )

            # ------------------------------------------------
            # Console output
            # ------------------------------------------------

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            print(
                f"Train Dice: {train_dice:.4f} | "
                f"Val Dice: {val_dice:.4f}"
            )

            print(
                f"Train IoU:  {train_iou:.4f} | "
                f"Val IoU:  {val_iou:.4f}"
            )

            print(
                f"Learning Rate: {current_lr:.2e}"
            )

            # ------------------------------------------------
            # MLflow metrics
            # ------------------------------------------------

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_dice": train_dice,
                    "val_dice": val_dice,
                    "train_iou": train_iou,
                    "val_iou": val_iou,
                    "learning_rate": current_lr,
                },
                step=epoch,
            )

            # ------------------------------------------------
            # Best model
            # ------------------------------------------------

            if val_dice > best_val_dice:

                best_val_dice = val_dice

                checkpoint_path = (
                    CHECKPOINT_DIR
                    / "best_model.pth"
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_dice": val_dice,
                        "val_iou": val_iou,
                    },
                    checkpoint_path,
                )

                mlflow.log_artifact(
                    str(checkpoint_path)
                )

                print(
                    f"✓ Best model saved "
                    f"(Dice: {val_dice:.4f})"
                )

        print(
            f"\nTraining complete."
        )

        print(
            f"Best validation Dice: "
            f"{best_val_dice:.4f}"
        )


if __name__ == "__main__":
    main()