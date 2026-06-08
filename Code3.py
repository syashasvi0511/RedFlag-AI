%%writefile app.py
import streamlit as st, torch, sys
sys.path.insert(0, ".")
from PIL import Image

st.set_page_config(page_title="Anaemia Screener", page_icon="👁", layout="centered")

@st.cache_resource
def get_model():
    # inline import so the app is self-contained
    import torch.nn as nn
    from torchvision import models
    ckpt = torch.load("models/anaemia_model.pt", map_location="cpu")
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(1280,128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128,2)
    )
    m.load_state_dict(ckpt["model_state_dict"])
    return m.eval(), ckpt.get("class_to_idx", {"healthy":0,"anaemic":1})

# paste inference functions inline
import torch, torch.nn.functional as F
from torchvision import transforms
import numpy as np, cv2

MEAN,STD,SZ = [0.485,0.456,0.406],[0.229,0.224,0.225],224
tf = transforms.Compose([transforms.Resize((SZ,SZ)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])

def predict(model, img, class_to_idx):
    ai = class_to_idx.get("anaemic",1)
    with torch.no_grad():
        probs = F.softmax(model(tf(img).unsqueeze(0)),dim=1)[0]
    ap = probs[ai].item()
    return dict(anaemic=round(ap,4), healthy=round(1-ap,4),
                risk="High" if ap>=.60 else ("Moderate" if ap>=.35 else "Low"),
                refer=ap>=.60, label="anaemic" if ap>=.5 else "healthy")

def gradcam(model, img):
    act,grd={},{}
    h1=model.features[-1].register_forward_hook(lambda m,i,o: act.update({"v":o.detach()}))
    h2=model.features[-1].register_full_backward_hook(lambda m,i,o: grd.update({"v":o[0].detach()}))
    t=tf(img).unsqueeze(0).requires_grad_(True)
    logits=model(t); model.zero_grad(); logits[0,logits.argmax()].backward()
    h1.remove(); h2.remove()
    w2=(grd["v"].mean(dim=(2,3),keepdim=True)*act["v"]).sum(dim=1).squeeze()
    cam=F.relu(w2).numpy(); cam=(cam-cam.min())/(cam.max()+1e-8)
    ww,hh=img.size
    hm=cv2.applyColorMap(np.uint8(255*cv2.resize(cam,(ww,hh))),cv2.COLORMAP_JET)
    return Image.fromarray(np.uint8(.45*cv2.cvtColor(hm,cv2.COLOR_BGR2RGB)+.55*np.array(img.convert("RGB"))))

# ── UI ───────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center'>👁 Anaemia Screener</h1>", unsafe_allow_html=True)
st.caption("CNN-based RBC image analysis · 96.3% sensitivity · For screening purposes only")
st.divider()

uploaded = st.file_uploader("Upload an RBC microscopy image", type=["png","jpg","jpeg"])
show_gc  = st.toggle("Show Grad-CAM heatmap", value=True)

COLORS = {"Low":"#1D9E75","Moderate":"#BA7517","High":"#A32D2D"}
BG     = {"Low":"#E1F5EE","Moderate":"#FAEEDA","High":"#FCEBEB"}

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    model, c2i = get_model()

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded image", use_container_width=True)
    with col2:
        with st.spinner("Analysing..."):
            r = predict(model, img, c2i)
        risk, c, bg = r["risk"], COLORS[r["risk"]], BG[r["risk"]]
        st.markdown(f"""
        <div style='background:{bg};border-radius:12px;padding:1.2rem'>
          <div style='color:{c};font-size:.8rem;font-weight:600;text-transform:uppercase'>Anaemia risk</div>
          <div style='color:{c};font-size:2rem;font-weight:700'>{risk}</div>
          <div style='font-size:.85rem;color:#444;margin-top:4px'>
            {"⚠️ Refer for blood test immediately." if r["refer"] else
             "Monitor. Consult doctor if symptoms persist." if risk=="Moderate" else
             "No significant indicators. Maintain iron-rich diet."}
          </div>
        </div>""", unsafe_allow_html=True)
        st.metric("Anaemic probability", f"{round(r['anaemic']*100)}%")
        st.progress(r["anaemic"])
        st.metric("Healthy probability",  f"{round(r['healthy']*100)}%")
        st.progress(r["healthy"])

    if show_gc:
        st.divider()
        st.subheader("Grad-CAM — model focus area")
        gc_col, _ = st.columns([1,1])
        with gc_col:
            st.image(gradcam(model, img), caption="Warmer = higher influence", use_container_width=True)

    st.divider()
    st.caption("Model: MobileNetV2 fine-tuned on AneRBC-II · Test accuracy: 73.8% · Sensitivity: 96.3% · Specificity: 51.3%")
else:
    st.markdown("""
    <div style='border:1.5px dashed #ccc;border-radius:12px;padding:3rem;text-align:center;color:#888'>
      <div style='font-size:2.5rem'>🔬</div>
      <p>Upload an RBC microscopy image to begin</p>
    </div>""", unsafe_allow_html=True)
