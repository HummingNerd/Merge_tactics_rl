# model_loader.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

def load_resnet(checkpoint_path, conf_thresh=0.5):
    """
    Loads model AND class names automatically from the checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load 'Smart' Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract Metadata
    if 'classes' in checkpoint:
        classes = checkpoint['classes']
        state_dict = checkpoint['state_dict']
    else:
        raise ValueError("❌ Checkpoint missing 'classes' key! Re-train using Script 1.")

    # Rebuild Architecture
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, len(classes))
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    # Define Transform
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return {
        "model": model,
        "classes": classes,
        "transform": tf,
        "device": device,
        "conf_thresh": conf_thresh
    }

def predict_crop(bundle, crop_bgr):
    """
    Input: OpenCV Image (BGR)
    Output: Label String, Confidence Float
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "EMPTY", 0.0

    # 1. Convert BGR to RGB (Fixes the color bug)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)

    # 2. Prepare Tensor
    img_t = bundle["transform"](pil_img).unsqueeze(0).to(bundle["device"])

    # 3. Predict
    with torch.no_grad():
        out = bundle["model"](img_t)
        probs = F.softmax(out, dim=1)
        conf, idx = probs.max(dim=1)

    conf_val = conf.item()
    label = bundle["classes"][idx.item()]

    if conf_val < bundle["conf_thresh"]:
        return "UNCERTAIN", conf_val

    return label, conf_val

if __name__ == "__main__":

    # 1. Load (No need to provide class list anymore!)
    troop_net = load_resnet("models/level_net.pth")

    # 2. Get image from screen
    img = cv2.imread("debug/5.png") 

    # 3. Predict
    label, conf = predict_crop(troop_net, img)
    print(f"I see: {label} ({conf:.2f})")