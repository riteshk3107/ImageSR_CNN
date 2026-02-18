"""
Flask UI for SRCNN: upload an image, see low-resolution (bicubic) vs SRCNN super-resolved.
"""
import os
import io
import base64

import numpy as np
import torch
from flask import Flask, render_template, request
from PIL import Image

from srcnn import ResidualSRCNN  # must match class name in srcnn.py


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), "srcnn.pth")
SCALE = 2


def load_model() -> bool:
    """Load SRCNN weights from srcnn.pth into global model."""
    global model
    if not os.path.isfile(MODEL_PATH):
        return False
    m = ResidualSRCNN().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    m.load_state_dict(state)
    m.eval()
    model = m
    return True


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_for_srcnn(pil_img: Image.Image, scale: int = SCALE):
    """
    Downscale Y by `scale`, bicubic upscale back -> low-res input for SRCNN.
    Returns (original_y, bicubic_y_for_model, tensor_input, cb, cr).
    """
    img = pil_img.convert("YCbCr")
    y, cb, cr = img.split()
    w, h = y.size

    lr_y = y.resize((w // scale, h // scale), Image.BICUBIC)
    bicubic_y = lr_y.resize((w, h), Image.BICUBIC)

    inp = np.array(bicubic_y).astype(np.float32) / 255.0
    inp = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(device)
    return y, bicubic_y, inp, cb, cr


def run_srcnn(inp: torch.Tensor) -> Image.Image:
    """Run SRCNN to predict residual and add it to bicubic input."""
    assert model is not None, "Model must be loaded before calling run_srcnn"
    with torch.no_grad():
        residual = model(inp)
        sr = inp + residual
    sr_np = sr.squeeze().cpu().numpy()
    sr_np = np.clip(sr_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(sr_np)


def merge_ycbcr(y: Image.Image, cb: Image.Image, cr: Image.Image) -> Image.Image:
    return Image.merge("YCbCr", [y, cb, cr]).convert("RGB")


def pil_to_base64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", result=None, error=None)

    if "image" not in request.files:
        return render_template("index.html", result=None, error="No image selected.")

    f = request.files["image"]
    if f.filename == "":
        return render_template("index.html", result=None, error="No image selected.")

    if not allowed_file(f.filename):
        return render_template(
            "index.html",
            result=None,
            error="Allowed formats: PNG, JPG, JPEG, BMP, WEBP.",
        )

    if model is None:
        return render_template(
            "index.html",
            result=None,
            error="Model not loaded. Train the model first (run srcnn.py) and save as srcnn.pth.",
        )

    try:
        pil_img = Image.open(f.stream).convert("RGB")
    except Exception as e:
        return render_template("index.html", result=None, error=f"Invalid image: {e}")

    try:
        original_y, bicubic_y, inp, cb, cr = preprocess_for_srcnn(pil_img)
        sr_y = run_srcnn(inp)

        original_rgb = merge_ycbcr(original_y, cb, cr)
        low_res_rgb = merge_ycbcr(bicubic_y, cb, cr)
        srcnn_rgb = merge_ycbcr(sr_y, cb, cr)

        original_b64 = pil_to_base64(original_rgb)
        low_res_b64 = pil_to_base64(low_res_rgb)
        srcnn_b64 = pil_to_base64(srcnn_rgb)

        return render_template(
            "index.html",
            result={"original": original_b64, "low_res": low_res_b64, "srcnn": srcnn_b64},
            error=None,
        )
    except Exception as e:
        return render_template("index.html", result=None, error=str(e))


if __name__ == "__main__":
    if load_model():
        print("SRCNN model loaded from srcnn.pth")
    else:
        print("Warning: srcnn.pth not found. Train the model first (run srcnn.py).")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)

