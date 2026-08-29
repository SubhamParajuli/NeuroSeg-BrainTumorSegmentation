from io import BytesIO
from pathlib import Path
import base64

import numpy as np
import tifffile
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from brain_seg.inference import BrainTumorInference


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CHECKPOINT = Path(
    "outputs/checkpoints/best_model.pth"
)

IMAGE_SIZE = 256
THRESHOLD = 0.60


# --------------------------------------------------
# Inference engine
# --------------------------------------------------

engine = BrainTumorInference(
    checkpoint_path=str(CHECKPOINT),
    image_size=IMAGE_SIZE,
    threshold=THRESHOLD,
)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="NeuroSeg Brain Tumor Segmentation API",
    description=(
        "CUDA-accelerated brain tumor segmentation "
        "using a ResNet18 encoder with U-Net decoder."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# Root
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "service": "NeuroSeg",
        "model": "ResNet18-UNet",
        "device": str(engine.device),
        "threshold": engine.threshold,
    }


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(engine.device),
        "cuda_available": str(
            engine.device
        ) == "cuda",
    }


# --------------------------------------------------
# Model information
# --------------------------------------------------

@app.get("/model/info")
def model_info():

    parameters = sum(
        p.numel()
        for p in engine.model.parameters()
    )

    return {
        "model": "ResNet18-UNet",
        "parameters": parameters,
        "input_size": [
            IMAGE_SIZE,
            IMAGE_SIZE,
        ],
        "threshold": THRESHOLD,
        "device": str(engine.device),
        "test_dice": 0.8611,
        "test_iou": 0.8321,
    }


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type.",
        )

    if not file.content_type.startswith(
        "image/"
    ):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    try:

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        # --------------------------------------------------
        # Read TIFF / image
        # --------------------------------------------------

        try:
            image = tifffile.imread(
                BytesIO(contents)
            )
        except Exception:

            image = np.array(
                Image.open(
                    BytesIO(contents)
                ).convert("RGB")
            )

        # --------------------------------------------------
        # Inference
        # --------------------------------------------------

        result = engine.predict(
            image
        )

        mask = result["mask"]

        # --------------------------------------------------
        # Resize original image
        # --------------------------------------------------

        original = engine.prepare_image(
            image
        )

        original_pil = Image.fromarray(
            original
        )

        original_pil = original_pil.resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        )

        original_array = np.array(
            original_pil
        )

        # --------------------------------------------------
        # Create overlay
        # --------------------------------------------------

        overlay = original_array.copy()

        tumor_region = mask == 1

        overlay[tumor_region] = (
            overlay[tumor_region] * 0.4
            + np.array([255, 0, 0]) * 0.6
        ).astype(np.uint8)

        # --------------------------------------------------
        # Encode overlay
        # --------------------------------------------------

        output = BytesIO()

        Image.fromarray(
            overlay
        ).save(
            output,
            format="PNG",
        )

        overlay_base64 = base64.b64encode(
            output.getvalue()
        ).decode("utf-8")

        # --------------------------------------------------
        # Response
        # --------------------------------------------------

        return JSONResponse(
            content={
                "tumor_detected": result[
                    "tumor_detected"
                ],
                "tumor_area_pixels": result[
                    "tumor_area_pixels"
                ],
                "tumor_percentage": round(
                    result[
                        "tumor_percentage"
                    ],
                    2,
                ),
                "inference_time_ms": round(
                    result[
                        "inference_time_ms"
                    ],
                    2,
                ),
                "threshold": THRESHOLD,
                "image_size": [
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                ],
                "model": "ResNet18-UNet",
                "device": str(
                    engine.device
                ),
                "overlay": overlay_base64,
            }
        )

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        )