from pathlib import Path

import matplotlib.pyplot as plt

from brain_seg.dataset import create_dataloaders


OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


train_loader, _, _ = create_dataloaders(
    data_dir="data/raw",
    batch_size=4,
    image_size=256,
)

images, masks = next(iter(train_loader))

fig, axes = plt.subplots(
    4,
    2,
    figsize=(8, 14),
)

for i in range(4):

    image = images[i].permute(1, 2, 0).numpy()
    mask = masks[i].squeeze(0).numpy()

    # Undo normalization
    image = image * 0.5 + 0.5
    image = image.clip(0, 1)

    axes[i, 0].imshow(image)
    axes[i, 0].set_title("MRI")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(mask, cmap="gray")
    axes[i, 1].set_title("Tumor Mask")
    axes[i, 1].axis("off")


plt.tight_layout()

output_path = OUTPUT_DIR / "dataset_samples.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
)

plt.show()

print(f"Saved visualization to: {output_path}")