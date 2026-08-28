from glob import glob
from pathlib import Path

import tifffile


files = glob("data/raw/**/*.tif", recursive=True)

images = [
    path for path in files
    if not Path(path).stem.endswith("_mask")
]

masks = [
    path for path in files
    if Path(path).stem.endswith("_mask")
]

print(f"Images: {len(images)}")
print(f"Masks:  {len(masks)}")

print("\nSample image shapes:")
for path in images[:10]:
    image = tifffile.imread(path)
    print(Path(path).name, image.shape, image.dtype)

print("\nSample mask shapes:")
for path in masks[:10]:
    mask = tifffile.imread(path)
    print(Path(path).name, mask.shape, mask.dtype)