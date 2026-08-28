import torch

from brain_seg.losses import BCEDiceLoss
from brain_seg.metrics import dice_score, iou_score


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # Fake model output
    logits = torch.randn(
        4,
        1,
        256,
        256,
        device=device,
    )

    # Fake binary masks
    targets = torch.randint(
        0,
        2,
        (
            4,
            1,
            256,
            256,
        ),
        device=device,
    ).float()

    loss_fn = BCEDiceLoss()

    loss = loss_fn(
        logits,
        targets,
    )

    dice = dice_score(
        logits,
        targets,
    )

    iou = iou_score(
        logits,
        targets,
    )

    print("Loss:", loss.item())
    print("Dice:", dice)
    print("IoU:", iou)


if __name__ == "__main__":
    main()