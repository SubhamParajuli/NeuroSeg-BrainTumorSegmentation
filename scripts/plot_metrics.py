import json
from pathlib import Path

import matplotlib.pyplot as plt


METRICS_FILE = Path(
    "outputs/evaluation/metrics.json"
)

OUTPUT_FILE = Path(
    "outputs/evaluation/metric_comparison.png"
)


with open(
    METRICS_FILE,
    "r",
    encoding="utf-8",
) as f:
    metrics = json.load(f)


names = [
    "Dice",
    "IoU",
    "Precision",
    "Recall",
    "Specificity",
    "Accuracy",
]

values = [
    metrics["dice"],
    metrics["iou"],
    metrics["precision"],
    metrics["recall"],
    metrics["specificity"],
    metrics["accuracy"],
]


plt.figure(figsize=(10, 6))

bars = plt.bar(
    names,
    values,
)

plt.ylim(0, 1.05)

plt.ylabel("Score")

plt.title(
    "NeuroSeg Test Set Performance"
)

for bar, value in zip(
    bars,
    values,
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.01,
        f"{value:.4f}",
        ha="center",
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
)

plt.close()

print(
    f"Saved: {OUTPUT_FILE}"
)