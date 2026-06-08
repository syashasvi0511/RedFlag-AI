import os, shutil, random, torch, torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms, datasets
from tqdm import tqdm
import copy

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = "AneRBC-II/AneRBC dataset a benchmark dataset for computer-aided anemia diagnosis using RBC images. httpsdoi.org10.1093databasebaae120/AneRBC_dataset/AneRBC-II"

SRC = {
    "anaemic": f"{BASE}/Anemic_individuals/RGB_segmented",
    "healthy":  f"{BASE}/Healthy_individuals/RGB_segmented",
}
DATA_DIR = Path("data_split")

# ── Build train/val/test split ─────────────────────────────────────────────
random.seed(42)
for cls, src in SRC.items():
    imgs = list(Path(src).glob("*.png"))
    random.shuffle(imgs)
    n = len(imgs)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    splits  = {"train": imgs[:n_train],
               "val":   imgs[n_train:n_train+n_val],
               "test":  imgs[n_train+n_val:]}
    for split, files in splits.items():
        dest = DATA_DIR / split / cls
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)
    print(f"{cls}: total={n}  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")

# ── Transforms ────────────────────────────────────────────────────────────
IMG_SIZE = 224
MEAN, STD = [0.485,0.456,0.406], [0.229,0.224,0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.ImageFolder(str(DATA_DIR/"train"), transform=train_tf)
val_ds   = datasets.ImageFolder(str(DATA_DIR/"val"),   transform=val_tf)
test_ds  = datasets.ImageFolder(str(DATA_DIR/"test"),  transform=val_tf)

print(f"\nClasses: {train_ds.classes}  (idx mapping: {train_ds.class_to_idx})")

# Weighted sampler to handle class imbalance
counts = np.bincount(train_ds.targets)
sample_weights = [1.0/counts[t] for t in train_ds.targets]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,   num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False,   num_workers=2, pin_memory=True)

# ── Model ─────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
for p in model.parameters():
    p.requires_grad = False

model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 2),
)
model = model.to(device)

# Penalise false negatives (anaemic missed) 2×
class_weights = torch.tensor([1.0, 2.0], device=device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# ── Training loop ─────────────────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    tp=fp=fn=tn=0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            tp += ((preds==1)&(labels==1)).sum().item()
            fp += ((preds==1)&(labels==0)).sum().item()
            fn += ((preds==0)&(labels==1)).sum().item()
            tn += ((preds==0)&(labels==0)).sum().item()
    n = tp+fp+fn+tn
    sens = tp/(tp+fn) if (tp+fn) else 0
    spec = tn/(tn+fp) if (tn+fp) else 0
    acc  = (tp+tn)/n if n else 0
    return dict(accuracy=acc, sensitivity=sens, specificity=spec)

best_sens, best_weights = 0.0, None

# Stage 1 — head only (10 epochs)
print("\n── Stage 1: training head ──")
opt1 = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
for epoch in range(10):
    model.train()
    for imgs, labels in tqdm(train_loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        opt1.zero_grad()
        criterion(model(imgs), labels).backward()
        opt1.step()
    m = evaluate(val_loader)
    print(f"  Epoch {epoch+1:02d}  acc={m['accuracy']:.3f}  sens={m['sensitivity']:.3f}  spec={m['specificity']:.3f}")
    if m['sensitivity'] > best_sens:
        best_sens = m['sensitivity']
        best_weights = copy.deepcopy(model.state_dict())

# Stage 2 — unfreeze top 30 layers (10 epochs)
print("\n── Stage 2: fine-tuning top layers ──")
model.load_state_dict(best_weights)
for layer in list(model.features.children())[-30:]:
    for p in layer.parameters(): p.requires_grad = True
for p in model.classifier.parameters(): p.requires_grad = True

opt2 = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
for epoch in range(10):
    model.train()
    for imgs, labels in tqdm(train_loader, leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        opt2.zero_grad()
        criterion(model(imgs), labels).backward()
        opt2.step()
    m = evaluate(val_loader)
    print(f"  Epoch {epoch+1:02d}  acc={m['accuracy']:.3f}  sens={m['sensitivity']:.3f}  spec={m['specificity']:.3f}")
    if m['sensitivity'] > best_sens:
        best_sens = m['sensitivity']
        best_weights = copy.deepcopy(model.state_dict())

# ── Final test eval ───────────────────────────────────────────────────────
model.load_state_dict(best_weights)
test_m = evaluate(test_loader)
print(f"\n── Test results ──")
for k, v in test_m.items():
    print(f"  {k}: {v:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
torch.save({
    "model_state_dict": best_weights,
    "classes": train_ds.classes,
    "class_to_idx": train_ds.class_to_idx,
    "test_metrics": test_m,
}, "models/anaemia_model.pt")
print("\nModel saved → models/anaemia_model.pt")
