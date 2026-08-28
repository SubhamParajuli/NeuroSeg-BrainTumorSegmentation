from brain_seg.dataset import create_dataloaders


train_loader, val_loader, test_loader = create_dataloaders(
    data_dir="data/raw",
    batch_size=4,
    image_size=256,
)

images, masks = next(iter(train_loader))

print("\n===== BATCH CHECK =====")
print("Images:")
print("  Shape:", images.shape)
print("  Min:", images.min().item())
print("  Max:", images.max().item())

print("\nMasks:")
print("  Shape:", masks.shape)
print("  Min:", masks.min().item())
print("  Max:", masks.max().item())