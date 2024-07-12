from PIL import Image
import torch
from torchvision import transforms
from models.utils import *
# library YOLOv8
from ultralytics import YOLO

# Load cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO()  # or create an empty model (new)

# Image
img = Image.open("../image/img.png")

# Inference
results = model(img)
results.show()  # display results

