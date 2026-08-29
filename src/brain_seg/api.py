from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
import tifffile

from brain_seg.dataset import get_val_transform
from brain_seg.model import UNet


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT = Path(
    "outputs/checkpoints/best_model.pth"
)

IMAGE_SIZE = 256
THRESHOLD = 0.60


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# --------------------------------------------------
# Load model
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

print("Model loaded successfully")


# --------------------------------------------------
# Validation preprocessing
# --------------------------------------------------

transform = get_val_transform(
    image_size=IMAGE_SIZE
)


def preprocess_image(
    image: Image.Image,
) -> torch.Tensor:

    # PIL → NumPy
    image = image.convert("RGB")

    image = np.array(
        image,
        dtype=np.uint8,
    )

    # USE EXACT SAME PREPROCESSING
    # AS VALIDATION DATASET
    transformed = transform(
        image=image
    )

    tensor = transformed["image"]

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    return tensor


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Brain Tumor Segmentation API",
    description=(
        "Brain tumor segmentation using "
        "a ResNet18 encoder with U-Net decoder."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Brain Tumor Segmentation API",
        "model": "ResNet18-UNet",
        "device": str(device),
        "threshold": THRESHOLD,
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(device),
    }


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):

    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:

        contents = await file.read()

        # Read TIFF exactly like the training dataset
        image = tifffile.imread(
            BytesIO(contents)
        )

        # Ensure RGB
        if image.ndim == 2:
            image = np.stack(
                [image, image, image],
                axis=-1,
            )

        if image.shape[-1] > 3:
            image = image[:, :, :3]

        image = image.astype(np.uint8)

        # EXACT same validation transform
        transformed = transform(
            image=image
        )

        tensor = (
            transformed["image"]
            .unsqueeze(0)
            .to(device)
        )
        # --------------------------------------------------
        # Inference
        # --------------------------------------------------

        with torch.no_grad():

            logits = model(tensor)

            probability = torch.sigmoid(
                logits
            )

            mask = (
                probability >= THRESHOLD
            ).float()

        # --------------------------------------------------
        # Convert prediction to PNG
        # --------------------------------------------------

        mask = (
            mask
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.uint8)
            * 255
        )

        output = BytesIO()

        Image.fromarray(
            mask
        ).save(
            output,
            format="PNG",
        )

        output.seek(0)

        return Response(
            content=output.getvalue(),
            media_type="image/png",
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        )