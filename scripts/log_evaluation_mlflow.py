import json
from pathlib import Path

import mlflow


METRICS_FILE = Path(
    "outputs/evaluation/metrics.json"
)

mlflow.set_experiment(
    "brain-tumor-segmentation"
)

with open(
    METRICS_FILE,
    "r",
    encoding="utf-8",
) as f:
    metrics = json.load(f)


with mlflow.start_run(
    run_name="final-test-evaluation"
):

    # Model information
    mlflow.log_param(
        "model",
        metrics["model"],
    )

    mlflow.log_param(
        "image_size",
        metrics["image_size"],
    )

    mlflow.log_param(
        "threshold",
        metrics["threshold"],
    )

    mlflow.log_param(
        "test_samples",
        metrics["test_samples"],
    )

    mlflow.log_param(
        "parameters",
        metrics["parameters"],
    )

    # Final evaluation metrics
    mlflow.log_metric(
        "test_dice",
        metrics["dice"],
    )

    mlflow.log_metric(
        "test_iou",
        metrics["iou"],
    )

    mlflow.log_metric(
        "test_precision",
        metrics["precision"],
    )

    mlflow.log_metric(
        "test_recall",
        metrics["recall"],
    )

    mlflow.log_metric(
        "test_specificity",
        metrics["specificity"],
    )

    mlflow.log_metric(
        "test_accuracy",
        metrics["accuracy"],
    )

    # Save evaluation artifacts
    mlflow.log_artifact(
        str(METRICS_FILE)
    )

    prediction_plot = Path(
        "outputs/evaluation/prediction_samples.png"
    )

    if prediction_plot.exists():
        mlflow.log_artifact(
            str(prediction_plot)
        )

    comparison_plot = Path(
        "outputs/evaluation/metric_comparison.png"
    )

    if comparison_plot.exists():
        mlflow.log_artifact(
            str(comparison_plot)
        )

    print("Final evaluation logged to MLflow.")