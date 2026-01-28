import torch
from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm

MODEL_PATH = "runs/detect/heridal_training/weights/best.pt"

INPUT_DIR = "dataset/images/test"
OUTPUT_DIR = "dataset/images/adversarial/fgsm_eps_0.05"

EPSILON = 0.05
DEVICE = "cpu"
IMG_SIZE = 640

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Učitaj model
model = YOLO(MODEL_PATH)
model.model.to(DEVICE)
model.model.eval()

def fgsm_attack(image, gradient, epsilon):
    sign_grad = gradient.sign()
    adv_image = image + epsilon * sign_grad
    adv_image = torch.clamp(adv_image, 0, 1)
    return adv_image

print("Generiram FGSM adversarial slike")

for img_name in tqdm(os.listdir(INPUT_DIR)):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(INPUT_DIR, img_name)

    # Učitaj sliku
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    img_tensor = torch.tensor(img_resized, dtype=torch.float32) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    img_tensor.requires_grad = True
    img_tensor = img_tensor.to(DEVICE)

    # Forward + loss
    results = model.model(img_tensor)
    loss = results[0].sum()

    # Backprop
    model.model.zero_grad()
    loss.backward()

    # FGSM
    adv_tensor = fgsm_attack(img_tensor, img_tensor.grad, EPSILON)

    # Spremi sliku
    adv_img = adv_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    adv_img = (adv_img * 255).astype("uint8")
    adv_img = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), adv_img)

print("FGSM napad gotov. Adversarial slike spremljene.")
