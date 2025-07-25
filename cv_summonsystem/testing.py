import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import time

def test_model(model, test_loader, device):
    model.eval()
    metric = MeanAveragePrecision()
    start_time = time.time()

    all_pred_scores = []
    all_true_labels = []

    pred_outputs = []
    true_targets = []

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)
            outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

            metric.update(outputs, targets)

            pred_outputs.extend(outputs)
            true_targets.extend(targets)

    end_time = time.time()
    elapsed = end_time - start_time
    fps = len(test_loader.dataset) / elapsed
    print(f"Inference Speed: {fps:.2f} FPS over {len(test_loader.dataset)} images")

    results = metric.compute()

    for preds, target in zip(pred_outputs, true_targets):
        pred_scores = preds['scores']
        true_boxes = target['boxes']

        for score in pred_scores:
            all_pred_scores.append(score.item())
            all_true_labels.append(1)  # predicted = positive

        missing = len(true_boxes) - len(pred_scores)
        for _ in range(max(0, missing)):
            all_pred_scores.append(0.0)
            all_true_labels.append(0)  # missed = false negative

    # Compute PR and F1
    precision, recall, thresholds = precision_recall_curve(all_true_labels, all_pred_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)

    # Plot: Precision–Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label="PR Curve")
    plt.scatter(recall[best_idx], precision[best_idx], color='red', label=f"Best F1={f1_scores[best_idx]:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.hist(all_pred_scores, bins=30, color='skyblue')
    plt.xlabel("Prediction Score")
    plt.ylabel("Count")
    plt.title("Distribution of Detection Scores")
    plt.grid(True)
    plt.show()

    # Print mAP Metrics
    print("Test mAP Results:")
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            v = v.item() if v.numel() == 1 else v
        print(f"  {k:15}: {v}")