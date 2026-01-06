from ultralytics import YOLO
import os
import pandas as pd
import torch

MODEL_PATH = "runs/detect/heridal_training/weights/best.pt"
DATA_YAML = "data.yaml"

TEST_IMG_DIR = "dataset/images/test"
TEST_LBL_DIR = "dataset/labels/test"

OUTPUT_CSV = "test_results_per_image.csv"

CONF_THRES = 0.25
IOU_THRES = 0.5
DEVICE = "cpu"

# Učitaj model
model = YOLO(MODEL_PATH)

print("Pokrećem evaluaciju (YOLO val)")
metrics = model.val(
    data=DATA_YAML,
    split="test",
    device=DEVICE
)

print("\nGLOBALNI REZULTATI")
print(f"mAP50:      {metrics.box.map50:.4f}")
print(f"mAP50-95:   {metrics.box.map:.4f}")
print(f"Precision:  {metrics.box.mp:.4f}")
print(f"Recall:     {metrics.box.mr:.4f}")

# PER-IMAGE ANALIZA

results = []

def load_gt_count(label_path):
    if not os.path.exists(label_path):
        return 0
    with open(label_path) as f:
        return len(f.readlines())

print("\nAnaliza po svakoj slici.")

for img_name in sorted(os.listdir(TEST_IMG_DIR)):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(TEST_IMG_DIR, img_name)
    label_path = os.path.join(TEST_LBL_DIR, img_name.replace(".jpg", ".txt"))

    # Ground truth
    gt_count = load_gt_count(label_path)

    # Predikcija
    preds = model.predict(
        source=img_path,
        conf=CONF_THRES,
        iou=IOU_THRES,
        device=DEVICE,
        verbose=False
    )

    pred_boxes = preds[0].boxes
    pred_count = len(pred_boxes)

    tp = min(gt_count, pred_count)
    fp = max(0, pred_count - gt_count)
    fn = max(0, gt_count - pred_count)

    results.append({
        "image": img_name,
        "ground_truth_humans": gt_count,
        "predicted_humans": pred_count,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    })

#Spremi CSV
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nRezultati spremljeni u {OUTPUT_CSV}")
