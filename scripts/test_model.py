import torch

from brain_seg.model import UNet


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    model = UNet().to(device)

    # Fake batch
    x = torch.randn(
        2,
        3,
        256,
        256,
        device=device,
    )

    with torch.no_grad():

        output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable_params:,}"
    )


if __name__ == "__main__":
    main()