import time

import numpy as np
import torch
from PIL import Image

from brain_seg.dataset import get_val_transform
from brain_seg.model import UNet


class BrainTumorInference:
    """Production inference wrapper for the brain tumor segmentation model."""

    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        threshold: float = 0.60,
    ):
        self.image_size = image_size
        self.threshold = threshold

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = UNet(
            in_channels=3,
            out_channels=1,
            pretrained=False,
        ).to(self.device)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.transform = get_val_transform(
            image_size=image_size
        )

    def prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Convert input image into RGB uint8 format."""

        if image.ndim == 2:
            image = np.stack(
                [image, image, image],
                axis=-1,
            )

        if image.ndim != 3:
            raise ValueError(
                f"Expected 2D or 3D image, got shape {image.shape}"
            )

        if image.shape[-1] > 3:
            image = image[:, :, :3]

        if image.shape[-1] != 3:
            raise ValueError(
                f"Expected RGB image, got shape {image.shape}"
            )

        return image.astype(np.uint8)

    def preprocess(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:

        image = self.prepare_image(image)

        transformed = self.transform(
            image=image
        )

        tensor = transformed["image"]

        return tensor.unsqueeze(0).to(
            self.device
        )

    @torch.inference_mode()
    def predict(
        self,
        image: np.ndarray,
    ):

        start = time.perf_counter()

        original_image = self.prepare_image(
            image
        )

        tensor = self.preprocess(
            original_image
        )

        logits = self.model(tensor)

        probability = torch.sigmoid(logits)

        mask = (
            probability >= self.threshold
        ).float()

        mask = (
            mask.squeeze()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        inference_time = (
            time.perf_counter() - start
        )

        tumor_pixels = int(mask.sum())

        total_pixels = int(mask.size)

        tumor_percentage = (
            tumor_pixels / total_pixels * 100
        )

        tumor_detected = tumor_pixels > 0

        return {
            "mask": mask,
            "probability": probability.squeeze()
            .cpu()
            .numpy(),
            "tumor_detected": tumor_detected,
            "tumor_area_pixels": tumor_pixels,
            "tumor_percentage": tumor_percentage,
            "inference_time_ms": inference_time * 1000,
        }