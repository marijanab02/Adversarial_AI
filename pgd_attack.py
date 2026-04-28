import torch
from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm

MODEL_PATH = "runs/detect/heridal_training/weights/best.pt"

INPUT_DIR = "dataset/images/test"
OUTPUT_DIR = "dataset/images/adversarial/pgd_eps_0.1_alpha_0.025_iters_15"

EPSILON = 0.1
ALPHA = 0.025     # step size
ITERATIONS = 15   # broj koraka (PGD!)

DEVICE = "cpu"
IMG_SIZE = 640

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)
model.model.to(DEVICE)
model.model.eval()

def pgd_attack(model, image, epsilon, alpha, iters):
    original = image.clone().detach()

    adv = image.clone().detach()

    for _ in range(iters):
        adv.requires_grad = True

        outputs = model.model(adv)
        loss = outputs[0].sum()

        model.model.zero_grad()
        loss.backward()

        # PGD step
        grad = adv.grad.sign()
        adv = adv + alpha * grad

        # Project back into epsilon ball
        delta = torch.clamp(adv - original, min=-epsilon, max=epsilon)
        adv = torch.clamp(original + delta, 0, 1).detach()

    return adv

print("Generiram PGD adversarial slike...")

for img_name in tqdm(os.listdir(INPUT_DIR)):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(INPUT_DIR, img_name)

    # Load image
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    img_tensor = torch.tensor(img_resized, dtype=torch.float32) / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # PGD attack
    adv_tensor = pgd_attack(model, img_tensor, EPSILON, ALPHA, ITERATIONS)

    # Save image
    adv_img = adv_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    adv_img = (adv_img * 255).astype("uint8")
    adv_img = cv2.cvtColor(adv_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), adv_img)

print("PGD napad gotov. Slike spremljene.")