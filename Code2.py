# Install Streamlit + gradcam + ngrok for demo
!pip install streamlit grad-cam pyngrok -q

# ── Grad-CAM + inference code ─────────────────────────────────────────────
import torch, torch.nn.functional as F
from torchvision import models, transforms
import numpy as np, cv2
from PIL import Image
import io, base64

MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]
IMG_SIZE  = 224

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def load_model(path="models/anaemia_model.pt"):
    ckpt = torch.load(path, map_location="cpu")
    m = models.mobilenet_v2(weights=None)
    import torch.nn as nn
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(1280,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,2)
    )
    m.load_state_dict(ckpt["model_state_dict"])
    return m.eval(), ckpt.get("class_to_idx", {"healthy":0,"anaemic":1})

def gradcam_overlay(model, pil_img):
    activations, gradients = {}, {}
    def fwd_hook(m,i,o): activations["v"] = o.detach()
    def bwd_hook(m,i,o): gradients["v"]   = o[0].detach()
    h1 = model.features[-1].register_forward_hook(fwd_hook)
    h2 = model.features[-1].register_full_backward_hook(bwd_hook)

    tensor = preprocess(pil_img).unsqueeze(0).requires_grad_(True)
    logits = model(tensor)
    model.zero_grad()
    logits[0, logits.argmax()].backward()

    h1.remove(); h2.remove()
    weights = gradients["v"].mean(dim=(2,3), keepdim=True)
    cam = F.relu((weights * activations["v"]).sum(dim=1).squeeze())
    cam = cam.numpy()
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    w, h = pil_img.size
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255*cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = np.uint8(0.45*heatmap + 0.55*np.array(pil_img.convert("RGB")))
    return Image.fromarray(blended)

def predict(model, pil_img, class_to_idx):
    anaemic_idx = class_to_idx.get("anaemic", 1)
    with torch.no_grad():
        probs = F.softmax(model(preprocess(pil_img).unsqueeze(0)), dim=1)[0]
    ap = probs[anaemic_idx].item()
    hp = 1 - ap
    label = "anaemic" if ap >= 0.5 else "healthy"
    risk  = "High" if ap >= 0.60 else ("Moderate" if ap >= 0.35 else "Low")
    refer = ap >= 0.60
    return dict(label=label, anaemic_prob=round(ap,4), healthy_prob=round(hp,4),
                risk=risk, refer=refer)

print("Inference code ready.")
