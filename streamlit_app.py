from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(
    page_title="Rice Leaf Image Classification Research Prototype",
    page_icon="🌾",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "model.pth"
MANIFEST_PATH = ROOT / "deployment_model_manifest.json"

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    MANIFEST = json.load(f)

CLASS_NAMES = MANIFEST["class_order"]
EXPECTED_CLASSES = ["bacterial_leaf_blight", "blast", "brown_spot", "hispa", "normal"]
assert CLASS_NAMES == EXPECTED_CLASSES
assert MANIFEST.get("test_set_used_for_selection") is False

DISPLAY_NAMES = {
    "bacterial_leaf_blight": "Bacterial leaf blight",
    "blast": "Blast",
    "brown_spot": "Brown spot",
    "hispa": "Hispa",
    "normal": "Normal",
}


def sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


class ECABlock(nn.Module):
    def __init__(self, channels=960, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)
        y = self.sigmoid(self.conv(y))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)


def build_sequential_eca(num_classes=5):
    model = models.mobilenet_v3_large(weights=None)
    model.features[16] = nn.Sequential(model.features[16], ECABlock(960, 3))
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model", "net", "best_model_state_dict"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict) and checkpoint and all(
        torch.is_tensor(v) for v in checkpoint.values()
    ):
        return checkpoint
    raise TypeError("No model state_dict was found.")


def clean_keys(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value
    return cleaned


@st.cache_resource(show_spinner="Loading the selected model…")
def load_model():
    observed_sha = sha256(CHECKPOINT_PATH)
    if observed_sha != MANIFEST["checkpoint_sha256"]:
        raise RuntimeError("Model SHA-256 does not match the deployment manifest.")

    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    checkpoint_classes = checkpoint.get("class_names") if isinstance(checkpoint, dict) else None
    if checkpoint_classes is not None:
        assert list(checkpoint_classes) == CLASS_NAMES
    state_dict = clean_keys(extract_state_dict(checkpoint))

    model = build_sequential_eca(num_classes=5)
    assert sum(p.numel() for p in model.parameters()) == 4_208_440
    model.load_state_dict(state_dict, strict=True)
    layout = "features[16]-sequential-ECA"
    model.eval()
    with torch.inference_mode():
        assert tuple(model(torch.zeros(1, 3, 224, 224)).shape) == (1, 5)
    return model, layout, observed_sha


MODEL, MODEL_LAYOUT, OBSERVED_SHA = load_model()

RESIZE = transforms.Resize((224, 224))

TO_TENSOR = transforms.ToTensor()
NORMALISE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def prepare_image(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    model_view = RESIZE(image)
    input_tensor = NORMALISE(TO_TENSOR(model_view)).unsqueeze(0)
    return image, model_view, input_tensor


def find_last_conv2d(module):
    layers = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
    if not layers:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return layers[-1]


GRADCAM_LAYER = find_last_conv2d(MODEL.features)


def gradcam_for_class(input_tensor, class_index):
    activations, gradients = [], []

    def hook(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = GRADCAM_LAYER.register_forward_hook(hook)
    try:
        MODEL.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = MODEL(input_tensor)
            logits[0, class_index].backward()
        weights = gradients[-1].detach().mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[-1].detach()).sum(dim=1))[0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        return cam.cpu().numpy()
    finally:
        handle.remove()


def make_overlay(original, cam):
    rgb = original.convert("RGB")
    heat = Image.fromarray(np.uint8(cam * 255), mode="L").resize(
        rgb.size, Image.Resampling.BILINEAR
    )
    heat_np = np.asarray(heat, dtype=np.float32) / 255.0
    colour = matplotlib.colormaps["jet"](heat_np)[..., :3]
    colour_img = Image.fromarray(np.uint8(colour * 255), mode="RGB")
    return Image.blend(rgb, colour_img, alpha=0.42)


def predict_leaf(image):
    _original, model_view, x = prepare_image(image)
    with torch.inference_mode():
        probabilities = torch.softmax(MODEL(x), dim=1)[0].cpu().numpy()
    predicted_index = int(np.argmax(probabilities))
    overlay = make_overlay(model_view, gradcam_for_class(x, predicted_index))
    return probabilities, predicted_index, model_view, overlay


st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
.notice {border-left: 5px solid #b45309; padding: 12px 16px; background: #fff7ed;
         border-radius: 4px; margin: 0.8rem 0 1.3rem 0;}
.small-note {color: #5f6368; font-size: 0.92rem;}
</style>
""", unsafe_allow_html=True)

st.title("🌾 Rice Leaf Image Classification — Research Prototype")
st.markdown(
    f"**Model:** ECA-MobileNetV3-Large · **Target:** five rice-leaf classes · "
    f"**Selected run:** seed {MANIFEST['seed']}, 30% target adaptation"
)
st.markdown(
    '<div class="notice"><b>Research prototype.</b> Field performance and probability '
    'calibration were not evaluated.</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Use one clear rice-leaf image. Avoid collages, screenshots, and images dominated "
    "by unrelated background objects."
)

uploaded = st.file_uploader(
    "Upload a rice-leaf image", type=["jpg", "jpeg", "png", "webp"]
)

if uploaded is None:
    st.info("Upload a rice-leaf image to run the selected five-class model.")
else:
    image_bytes = uploaded.getvalue()
    upload_id = hashlib.sha256(image_bytes).hexdigest()
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be read as an image. Please choose another file.")
        st.stop()
    if st.session_state.get("upload_id") != upload_id:
        st.session_state.pop("prediction_result", None)
        st.session_state["upload_id"] = upload_id

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.image(image, caption="Uploaded rice-leaf image", use_container_width=True)
        analyse = st.button("Analyse image", type="primary", use_container_width=True)

    if analyse:
        with st.spinner("Analysing image and generating Grad-CAM…"):
            st.session_state["prediction_result"] = predict_leaf(image)

    result = st.session_state.get("prediction_result")
    with right:
        if result is None:
            st.info("Select **Analyse image** to view the prediction.")
        else:
            probabilities, predicted_index, _model_view, _overlay = result
            predicted_name = DISPLAY_NAMES[CLASS_NAMES[predicted_index]]
            st.subheader(f"Predicted class: {predicted_name}")
            st.metric(
                "Top-class softmax probability",
                f"{probabilities[predicted_index] * 100:.2f}%",
            )
            order = np.argsort(probabilities)[::-1]
            for rank, idx in enumerate(order, start=1):
                st.markdown(
                    f"{rank}. **{DISPLAY_NAMES[CLASS_NAMES[idx]]}** — "
                    f"{probabilities[idx] * 100:.2f}%"
                )
            st.caption("The displayed probabilities have not been calibrated.")

            chart_df = pd.DataFrame({
                "Class": [DISPLAY_NAMES[name] for name in CLASS_NAMES],
                "Probability (%)": probabilities * 100,
            }).sort_values("Probability (%)", ascending=True)
            fig, ax = plt.subplots(figsize=(7.0, 3.3))
            colors = ["#f59e0b" if x == predicted_name else "#9ca3af" for x in chart_df["Class"]]
            ax.barh(chart_df["Class"], chart_df["Probability (%)"], color=colors)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Class probabilities")
            ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    if result is not None:
        st.subheader("Model input and Grad-CAM")
        crop_col, cam_col = st.columns(2)
        with crop_col:
            st.image(
                result[2],
                caption="Model input after direct resize (224 × 224)",
                use_container_width=True,
            )
        with cam_col:
            st.image(
                result[3],
                caption="Grad-CAM attention overlay on the model input",
                use_container_width=True,
            )
        st.caption(
            "Warm colours indicate image regions that contributed more strongly to the "
            "selected class. The map is an explanatory aid, not a lesion boundary."
        )

with st.sidebar:
    st.header("Model information")
    st.write(f"Selection seed: **{MANIFEST['seed']}**")
    st.write(f"Best validation Macro-F1: **{MANIFEST['best_validation_macro_f1']:.6f}**")
    st.write(f"Architecture layout: `{MODEL_LAYOUT}`")
    st.write(f"Model SHA-256: `{OBSERVED_SHA[:16]}…`")
    st.caption("The final test set was not used to select this deployment checkpoint.")
