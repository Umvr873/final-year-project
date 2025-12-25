import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
from PIL import Image
import numpy as np
import matplotlib.cm as cm
import os
import urllib.request

# ========================= MODEL DOWNLOAD =========================
MODEL_URL = "https://github.com/Umvr873/final-year-project/releases/download/1.0/best_model.pth"
MODEL_PATH = "best_model.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ========================= CONFIG =========================
class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
confidence_threshold = 0.70

suggestions = {
    "COVID": (
        "1. Initiate appropriate infection prevention and control measures, including patient isolation where indicated.\n"
        "2. Recommend confirmatory laboratory testing for SARS-CoV-2 (e.g., RT-PCR or approved antigen tests).\n"
        "3. Perform clinical assessment to evaluate disease severity, including respiratory rate and oxygen saturation.\n"
        "4. Provide supportive care according to current clinical guidelines and escalate care if hypoxia or clinical deterioration is observed.\n"
        "5. Consider follow-up chest imaging if symptoms worsen or fail to improve."
    ),
    "Normal": (
        "1. No radiographic evidence of acute pulmonary pathology identified on the current image.\n"
        "2. Clinical correlation is advised, as early or mild disease may not be radiographically apparent.\n"
        "3. Continue routine clinical monitoring and standard preventive healthcare practices.\n"
        "4. Re-evaluation is recommended if respiratory symptoms persist or progress.\n"
        "5. No immediate imaging-based intervention is required."
    ),
    "Viral Pneumonia": (
        "1. Conduct further clinical evaluation to determine the likely viral etiology, including appropriate laboratory investigations.\n"
        "2. Initiate supportive management in accordance with established treatment protocols.\n"
        "3. Monitor respiratory function and oxygenation status, particularly in high-risk patients.\n"
        "4. Assess for potential complications or secondary bacterial infection if clinical status worsens.\n"
        "5. Consider follow-up imaging to evaluate treatment response when clinically indicated."
    ),
    "Lung_Opacity": (
        "1. Recommend additional imaging, such as high-resolution computed tomography (CT), for further characterization of the opacity.\n"
        "2. Correlate radiographic findings with clinical history and laboratory results.\n"
        "3. Consider referral to a pulmonologist for comprehensive evaluation.\n"
        "4. Differential diagnosis may include infectious, inflammatory, or neoplastic processes.\n"
        "5. Plan follow-up imaging based on the suspected underlying etiology and clinical progression."
    )
}

explanation = {
    "COVID": "The model identified bilateral pulmonary regions with increased attenuation patterns consistent with ground-glass opacities. Clinical correlation is required.",
    "Lung_Opacity": "The model detected areas of increased radiographic density. Further evaluation is needed.",
    "Viral Pneumonia": "Patchy heterogeneous opacities detected, suggestive of viral pneumonia. Correlate with clinical and lab data.",
    "Normal": "No significant radiographic abnormalities detected. Lung fields appear clear."
}

# ========================= LOAD MODEL =========================
@st.cache_resource
def load_model():
    model = timm.create_model("densenet121", pretrained=False, num_classes=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ========================= GRAD-CAM HOOKS =========================
activations, gradients = [], []

def forward_hook(module, input, output):
    activations.append(output)

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

model.features.denseblock4.register_forward_hook(forward_hook)
model.features.denseblock4.register_full_backward_hook(backward_hook)

# ========================= TRANSFORM =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========================= GRAD-CAM FUNCTION =========================
def generate_gradcam(model, image_tensor, class_idx):
    activations.clear()
    gradients.clear()
    output = model(image_tensor)
    model.zero_grad()
    output[0, class_idx].backward()

    act = activations[0].squeeze(0)
    grad = gradients[0].squeeze(0)
    pooled_grad = torch.mean(grad, dim=(1, 2))
    for i in range(act.shape[0]):
        act[i] *= pooled_grad[i]

    heatmap = torch.mean(act, dim=0).detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max() + 1e-8
    return heatmap

# ========================= STREAMLIT UI =========================
st.title("🩻🧠 Chest X-ray Disease Classifier")

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg","jpeg","png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        # Quick X-ray grayscale validation
        img_np = np.array(image)
        grayscale_ratio = np.mean(np.abs(img_np[:,:,0] - img_np[:,:,1]))

        if grayscale_ratio > 30:
            st.error("🚫 This does not appear to be a valid chest X-ray.")
        else:
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
            img_tensor = transform(image).unsqueeze(0)
            img_tensor.requires_grad_()

            outputs = model(img_tensor)
            probs = F.softmax(outputs[0], dim=0)
            pred_idx = torch.argmax(probs).item()
            pred_conf = probs[pred_idx].item()

            st.subheader("🔍 Prediction Probabilities")
            for i, p in enumerate(probs):
                st.write(f"{class_names[i]}: {p.item()*100:.2f}%")

            if pred_conf >= confidence_threshold:
                pred_class = class_names[pred_idx]
                st.success(f"🧪 Most likely diagnosis: **{pred_class}** ({pred_conf*100:.2f}%)")
                st.subheader("📋 Suggested Medical Steps")
                st.info(suggestions[pred_class])
                st.subheader("🌡️ Grad-CAM Heatmap")
                heatmap = generate_gradcam(model, img_tensor, pred_idx)

                heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)
                heatmap_img = np.array(heatmap_img)
                image_np = np.array(image).astype(np.float32)/255.0
                heatmap_color = cm.jet(heatmap_img/255.0)[..., :3]
                blended = np.clip(0.6*image_np + 0.4*heatmap_color, 0, 1)
                st.image(blended, caption="Model Attention Map", use_container_width=True)

                st.subheader("🧠 Model Explanation")
                st.write(explanation[pred_class])
            else:
                st.warning(f"⚠️ Low confidence prediction ({pred_conf*100:.2f}%).")
                st.info("Try uploading a clearer chest X-ray image.")

            st.caption("⚠️ Research use only — not for clinical diagnosis")

    except Exception as e:
        st.error(f"Unexpected error: {e}")
