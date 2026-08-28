import torch


def dice_score(
    logits,
    targets,
    threshold=0.5,
    smooth=1e-6,
):
    """
    Calculate Dice score for binary segmentation.
    """

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > threshold
    ).float()

    predictions = predictions.contiguous().view(
        predictions.size(0),
        -1,
    )

    targets = targets.contiguous().view(
        targets.size(0),
        -1,
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
    """
    Calculate IoU score for binary segmentation.
    """

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities > threshold
    ).float()

    predictions = predictions.contiguous().view(
        predictions.size(0),
        -1,
    )

    targets = targets.contiguous().view(
        targets.size(0),
        -1,
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