import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import os
import cv2
import easyocr
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse

# ----------- OCR & Regex Setup -----------
reader = easyocr.Reader(['en'], gpu=False)
pattern = r'\b[A-Z]{1,3}[0-9]{1,4}[A-Z]?\b'
summons_db = ["RQ9228", "VBE7951", "BJG2133", "VAG3627", "WXB8898"]


def run_ocr_on_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Run OCR on original image
    results = reader.readtext(thresh)

    # Combine all detected texts
    plate_number = " ".join([res[1] for res in results])

    # Clean and regex match
    clean_plate = re.sub(r'\s+', '', plate_number).upper()
    match = re.search(pattern, clean_plate)

    return match.group() if match else "Invalid"

# ----------- Load Model -----------
def load_model(checkpoint_path, device):
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print("Loaded best model.")
    return model

# ----------- Main Detection Function -----------
def detect_and_ocr(image_path, model, device, score_threshold=0.5):
    print(f"Running inference on {image_path}...")
    orig = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(orig).to(device)

    with torch.no_grad():
        output = model([img_tensor])[0]

    pred_boxes = output["boxes"].cpu()
    pred_scores = output["scores"].cpu()

    orig_cv2 = cv2.cvtColor(np.array(orig), cv2.COLOR_RGB2BGR)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(orig)
    found_any = False

    for box, score in zip(pred_boxes, pred_scores):
        if score >= score_threshold:
            found_any = True
            x1, y1, x2, y2 = map(int, box.tolist())
            cropped = orig_cv2[y1:y2, x1:x2]  # crop using OpenCV

            matched_plate = run_ocr_on_image(cropped)

            # Check against summons DB
            if matched_plate != "Invalid":
                if matched_plate in summons_db:
                    label = f"Overdue: {matched_plate}"
                else:
                    label = matched_plate
            else:
                label = "Invalid"

            # Draw box and label
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1 - 10, label, color='red', fontsize=12, weight='bold')

    if not found_any:
        print("No predictions above threshold.")

    plt.title("Final Detection with OCR Labels")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ----------- Entry Point -----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="Folder containing images")
    parser.add_argument("--model", type=str, default="best_model.pth", help="Path to best model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    supported_ext = ('.jpg', '.jpeg', '.png')
    images = sorted([
        os.path.join(args.folder, f)
        for f in os.listdir(args.folder)
        if f.lower().endswith(supported_ext)
    ])

    if not images:
        print("No valid image files found in the folder.")
    else:
        for img_path in images:
            detect_and_ocr(img_path, model, device, args.threshold)
