import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH = "runs/detect/heridal_training/weights/best.pt"
IMAGE_NAME = "train_ZRI_3016_JPG.rf.5c569a2f8126ecd868a8fd187719a7b9.jpg" 
IMG_DIR = "dataset/images/test"
LBL_DIR = "dataset/labels/test"
OUTPUT_DIR = "graphic_comparison"

CONF_THRES = 0.25
DEVICE = "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

img_path = os.path.join(IMG_DIR, IMAGE_NAME)
label_path = os.path.join(LBL_DIR, IMAGE_NAME.replace(".jpg", ".txt"))

# Učitaj sliku
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

gt_img = img.copy()

if os.path.exists(label_path):
    with open(label_path) as f:
        for line in f:
            _, x, y, bw, bh = map(float, line.split())

            xmin = int((x - bw / 2) * w)
            ymin = int((y - bh / 2) * h)
            xmax = int((x + bw / 2) * w)
            ymax = int((y + bh / 2) * h)

            cv2.rectangle(gt_img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

# PREDICTIONS
model = YOLO(MODEL_PATH)

pred_img = img.copy()
results = model.predict(
    source=img_path,
    conf=CONF_THRES,
    device=DEVICE,
    verbose=False
)

for box in results[0].boxes.xyxy.cpu().numpy():
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(pred_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# PLOT
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(gt_img)
axes[0].set_title("Ground Truth (Crveno)")
axes[0].axis("off")

axes[1].imshow(pred_img)
axes[1].set_title("YOLOv8 Predikcija (Zeleno)")
axes[1].axis("off")

output_path = os.path.join(
    OUTPUT_DIR,
    f"comparison_{os.path.splitext(IMAGE_NAME)[0]}.png"
)

plt.tight_layout()
plt.savefig(output_path, dpi=200)
plt.close()

print(f"Spremljena usporedba: {output_path}")
