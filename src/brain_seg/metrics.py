import torch


def _prepare_predictions(logits, targets, threshold=0.5):
    """Convert model logits and masks into flattened binary tensors."""

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= threshold
    ).float()

    predictions = predictions.contiguous().view(
        predictions.size(0),
        -1,
    )

    targets = targets.contiguous().view(
        targets.size(0),
        -1,
    )

    return predictions, targets


def dice_score(
    logits,
    targets,
    threshold=0.5,
    smooth=1e-6,
):
    """Calculate mean Dice score."""

    predictions, targets = _prepare_predictions(
        logits,
        targets,
        threshold,
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection + smooth
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + smooth
    )

    return dice.mean().item()


def iou_score(
    logits,
    targets,
    threshold=0.5,
    smooth=1e-6,
):
    """Calculate mean Intersection over Union."""

    predictions, targets = _prepare_predictions(
        logits,
        targets,
        threshold,
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum(dim=1)

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.mean().item()


def segmentation_metrics(
    logits,
    targets,
    threshold=0.5,
):
    """
    Calculate comprehensive binary segmentation metrics.
    """

    predictions, targets = _prepare_predictions(
        logits,
        targets,
        threshold,
    )

    tp = (
        predictions * targets
    ).sum(dim=1)

    fp = (
        predictions * (1 - targets)
    ).sum(dim=1)

    fn = (
        (1 - predictions) * targets
    ).sum(dim=1)

    tn = (
        (1 - predictions)
        * (1 - targets)
    ).sum(dim=1)

    smooth = 1e-6

    dice = (
        2 * tp + smooth
    ) / (
        2 * tp + fp + fn + smooth
    )

    iou = (
        tp + smooth
    ) / (
        tp + fp + fn + smooth
    )

    precision = (
        tp + smooth
    ) / (
        tp + fp + smooth
    )

    recall = (
        tp + smooth
    ) / (
        tp + fn + smooth
    )

    specificity = (
        tn + smooth
    ) / (
        tn + fp + smooth
    )

    accuracy = (
        tp + tn + smooth
    ) / (
        tp + tn + fp + fn + smooth
    )

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "specificity": specificity.mean().item(),
        "accuracy": accuracy.mean().item(),
    }