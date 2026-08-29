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

import base64
import time

from fastapi.responses import JSONResponse

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
        start_time = time.perf_counter()

        contents = await file.read()

        # --------------------------------------------------
        # Read image
        # --------------------------------------------------

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

        original_image = image.copy()

        # --------------------------------------------------
        # Same preprocessing as validation
        # --------------------------------------------------

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
        # Convert mask
        # --------------------------------------------------

        mask = (
            mask
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        # --------------------------------------------------
        # Tumor detection
        # --------------------------------------------------

        tumor_pixels = int(
            mask.sum()
        )

        total_pixels = int(
            mask.size
        )

        tumor_detected = (
            tumor_pixels > 0
        )

        tumor_percentage = (
            tumor_pixels
            / total_pixels
            * 100
        )

        # --------------------------------------------------
        # Resize original image to match mask
        # --------------------------------------------------

        original_pil = Image.fromarray(
            original_image
        )

        original_pil = original_pil.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        original_array = np.array(
            original_pil
        )

        # --------------------------------------------------
        # Create overlay
        # --------------------------------------------------

        overlay = original_array.copy()

        # Highlight predicted tumor
        # using a red overlay
        overlay[mask == 1] = (
            overlay[mask == 1] * 0.4
            + np.array([255, 0, 0]) * 0.6
        ).astype(np.uint8)

        # --------------------------------------------------
        # Convert overlay to PNG
        # --------------------------------------------------

        output = BytesIO()

        Image.fromarray(
            overlay
        ).save(
            output,
            format="PNG",
        )

        image_base64 = base64.b64encode(
            output.getvalue()
        ).decode("utf-8")

        inference_time = (
            time.perf_counter()
            - start_time
        )

        # --------------------------------------------------
        # JSON response
        # --------------------------------------------------

        return JSONResponse(
            content={
                "tumor_detected": tumor_detected,
                "tumor_area_pixels": tumor_pixels,
                "tumor_percentage": round(
                    tumor_percentage,
                    2,
                ),
                "inference_time_ms": round(
                    inference_time * 1000,
                    2,
                ),
                "threshold": THRESHOLD,
                "overlay": image_base64,
            }
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        )