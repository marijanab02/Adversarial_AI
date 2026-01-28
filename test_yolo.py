from ultralytics import YOLO
import os
import pandas as pd
import torch

MODEL_PATH = "runs/detect/heridal_training_FGSM/weights/best.pt"
DATA_YAML = "data_fgsm.yaml"

TEST_IMG_DIR = "dataset/images/adversarial/fgsm_eps_0.005"
TEST_LBL_DIR = "dataset/labels/adversarial/fgsm_eps_0.005"

OUTPUT_CSV = "results_after_adversial_training/test_results_per_image_fgsm_new_0.005.csv"
OUTPUT_GLOBAL_CSV = "results_after_adversial_training/global_results_fgsm_new_0.005.csv"

CONF_THRES = 0.25
IOU_THRES = 0.5
DEVICE = "cpu"

# Učitaj model
model = YOLO(MODEL_PATH)

# Globalni rezultati
print("Pokrećem evaluaciju (YOLO val)")
metrics = model.val(
    data=DATA_YAML,
    split="test",
    device=DEVICE
)

global_results = {
    "mAP50": round(metrics.box.map50, 4),
    "mAP50-95": round(metrics.box.map, 4),
    "Precision": round(metrics.box.mp, 4),
    "Recall": round(metrics.box.mr, 4)
}

print("\nGLOBALNI REZULTATI")
for k, v in global_results.items():
    print(f"{k}: {v}")

# Spremi globalne rezultate u CSV
df_global = pd.DataFrame([global_results])
df_global.to_csv(OUTPUT_GLOBAL_CSV, index=False)
print(f"Globalni rezultati spremljeni u {OUTPUT_GLOBAL_CSV}")

#Per-image analiza

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

    gt_count = load_gt_count(label_path)

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

# Spremi CSV po slici
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Per-image rezultati spremljeni u {OUTPUT_CSV}")
