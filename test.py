import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Residual SRCNN (must match srcnn.py) ----------------
class ResidualSRCNN(nn.Module):
    """
    5-layer residual SRCNN used for inference.
    Kernel sizes: [9, 5, 5, 5, 5], channels: 1 -> 64 -> 64 -> 32 -> 32 -> 1,
    receptive field: 25x25. The network outputs a residual which is added to the bicubic input.
    """
    def __init__(self):
        super(ResidualSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual = self.conv5(out)
        return residual
    

# ---------------- LOAD MODEL ----------------
def load_model(path="srcnn.pth"):
    model = ResidualSRCNN()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

# ---------------- PSNR ----------------
def calculate_psnr(img1, img2):
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

# ---------------- PREPROCESS ----------------
def preprocess_image(path, scale=2):
    img = Image.open(path).convert("YCbCr")
    y, cb, cr = img.split()

    lr = y.resize((y.width // scale, y.height // scale), Image.BICUBIC)
    bicubic = lr.resize((y.width, y.height), Image.BICUBIC)

    inp = np.array(bicubic).astype(np.float32) / 255.0
    inp = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0)

    return y, bicubic, inp, cb, cr

def tensor_to_image(t):
    t = t.squeeze().detach().numpy()
    t = np.clip(t * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(t)

def merge_ycbcr(y, cb, cr):
    return Image.merge("YCbCr", [y, cb, cr]).convert("RGB")

# ---------------- TEST ----------------
def test_and_compare(model, img_path):
    y, bicubic_y, inp, cb, cr = preprocess_image(img_path)

    with torch.no_grad():
        residual = model(inp)
        sr = inp + residual

    sr_y = tensor_to_image(sr)

    orig_rgb = merge_ycbcr(y, cb, cr)
    bicubic_rgb = merge_ycbcr(bicubic_y, cb, cr)
    sr_rgb = merge_ycbcr(sr_y, cb, cr)

    psnr_b = calculate_psnr(y, bicubic_y)
    psnr_s = calculate_psnr(y, sr_y)

    plt.figure(figsize=(15,5))
    titles = [
        "Original",
        f"Bicubic PSNR: {psnr_b:.2f} dB",
        f"SRCNN PSNR: {psnr_s:.2f} dB"
    ]
    images = [orig_rgb, bicubic_rgb, sr_rgb]

    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")

    plt.show()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    model = load_model("srcnn.pth")
    test_and_compare(model, "IWA015691_500px.jpg")
