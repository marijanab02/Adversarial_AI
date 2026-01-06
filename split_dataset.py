import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SOURCE_DIR = "train"
CSV_PATH = os.path.join(SOURCE_DIR, "annotations.csv")
BASE_DIR = "dataset"
IMG_OUT = os.path.join(BASE_DIR, "images")
LBL_OUT = os.path.join(BASE_DIR, "labels")

SPLIT = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

CLASS_MAP = {
    "human": 0
}

# Kreiraj direktorije
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(IMG_OUT, split), exist_ok=True)
    os.makedirs(os.path.join(LBL_OUT, split), exist_ok=True)

# Učitaj CSV
df = pd.read_csv(CSV_PATH)

# Dohvati jedinstvene slike
images = df["filename"].unique()

# Podjela
train_imgs, temp_imgs = train_test_split(
    images, test_size=1 - SPLIT["train"], random_state=42
)

val_imgs, test_imgs = train_test_split(
    temp_imgs,
    test_size=SPLIT["test"] / (SPLIT["val"] + SPLIT["test"]),
    random_state=42
)

splits = {
    "train": train_imgs,
    "val": val_imgs,
    "test": test_imgs
}

# Funkcija za YOLO format
def convert_to_yolo(row):
    x_center = (row.xmin + row.xmax) / 2 / row.width
    y_center = (row.ymin + row.ymax) / 2 / row.height
    w = (row.xmax - row.xmin) / row.width
    h = (row.ymax - row.ymin) / row.height
    return f"{CLASS_MAP[row['class']]} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"

# Obrada po splitovima
for split, img_list in splits.items():
    print(f"Processing {split} set...")

    for img in tqdm(img_list):
        src_img = os.path.join(SOURCE_DIR, img)
        dst_img = os.path.join(IMG_OUT, split, img)

        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)

        label_path = os.path.join(LBL_OUT, split, img.replace(".jpg", ".txt"))

        img_rows = df[df["filename"] == img]

        with open(label_path, "w") as f:
            for _, row in img_rows.iterrows():
                f.write(convert_to_yolo(row) + "\n")

print("Dataset je spreman za YOLOv8!")
