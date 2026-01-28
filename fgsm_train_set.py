import os
import cv2
import torch
import random
import shutil
from tqdm import tqdm
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "runs/detect/heridal_training/weights/best.pt"

ORIG_IMG_DIR = "dataset/images/train"
ORIG_LBL_DIR = "dataset/labels/train"

OUT_IMG_DIR = "dataset/adversarial_train/images"
OUT_LBL_DIR = "dataset/adversarial_train/labels"

EPSILON = 0.02
IMG_SIZE = 640
DEVICE = "cpu"
FGSM_RATIO = 0.3   # 30% FGSM
# ----------------------------------------

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)
model.model.to(DEVICE)
model.model.eval()

def fgsm_attack(image, gradient, epsilon):
    signed_grad = gradient.sign()
    adv_image = image + epsilon * signed_grad
    return torch.clamp(adv_image, 0, 1)

# Load images
images = [f for f in os.listdir(ORIG_IMG_DIR) if f.endswith(".jpg")]
random.shuffle(images)

split_idx = int(len(images) * (1 - FGSM_RATIO))
normal_imgs = images[:split_idx]
fgsm_imgs = images[split_idx:]

print(f"Normal images: {len(normal_imgs)}")
print(f"FGSM images:   {len(fgsm_imgs)}")
print(f"TOTAL:         {len(images)}")

# -------- COPY NORMAL IMAGES --------
print("\nKopiram normalne slike...")
for img in tqdm(normal_imgs):
    shutil.copy(
        os.path.join(ORIG_IMG_DIR, img),
        os.path.join(OUT_IMG_DIR, img)
    )
    shutil.copy(
        os.path.join(ORIG_LBL_DIR, img.replace(".jpg", ".txt")),
        os.path.join(OUT_LBL_DIR, img.replace(".jpg", ".txt"))
    )

# -------- GENERATE FGSM IMAGES --------
print("\nGeneriram FGSM slike...")

for img_name in tqdm(fgsm_imgs):
    img_path = os.path.join(ORIG_IMG_DIR, img_name)

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    img_tensor = torch.tensor(img_resized, dtype=torch.float32) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad = True

    # Forward
    outputs = model.model(img_tensor)
    loss = outputs[0].sum()

    model.model.zero_grad()
    loss.backward()

    adv_tensor = fgsm_attack(img_tensor, img_tensor.grad, EPSILON)

    adv_img = adv_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    adv_img = (adv_img * 255).astype("uint8")
    adv_img = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(OUT_IMG_DIR, img_name), adv_img)

    # Copy label (ISTA)
    shutil.copy(
        os.path.join(ORIG_LBL_DIR, img_name.replace(".jpg", ".txt")),
        os.path.join(OUT_LBL_DIR, img_name.replace(".jpg", ".txt"))
    )

print("\nFGSM 70/30 TRAIN SET USPJEŠNO GENERIRAN")
