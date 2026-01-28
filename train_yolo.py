from ultralytics import YOLO
import os

DATA_YAML = "data_fgsm_train.yaml"

MODEL = "yolov8n.pt"

EPOCHS = 20

BATCH_SIZE = 4

DEVICE = "cpu"

if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(f"{DATA_YAML} nije pronađen. Provjeri putanju.")

model = YOLO(MODEL)

# Pokretanje treniranja
print("Pokrećem treniranje YOLOv8 na CPU...")
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    device=DEVICE,
    imgsz=640,      
    save=True,       
    name="heridal_training_FGSM"
)

print("Treniranje završeno!")
