import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import glob
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- SRCNN MODEL (upgraded: deeper + residual learning) ----------------
class SRCNN(nn.Module):
    """Upgraded SRCNN: 5 conv layers, more channels, predicts residual (detail) over bicubic."""
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x

# ---------------- DATASET ----------------
class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, patch_size=96, scale=2):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        self.patch_size = patch_size
        self.scale = scale

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_paths[idx]).convert("YCbCr")
        y, _, _ = hr_img.split()

        hr_w, hr_h = y.size
        crop_size = self.patch_size * self.scale

        x = random.randint(0, hr_w - crop_size)
        y_pos = random.randint(0, hr_h - crop_size)

        hr_crop = y.crop((x, y_pos, x + crop_size, y_pos + crop_size))
        lr = hr_crop.resize((self.patch_size, self.patch_size), Image.BICUBIC)
        lr_up = lr.resize((crop_size, crop_size), Image.BICUBIC)

        hr = np.array(hr_crop).astype(np.float32) / 255.0
        lr = np.array(lr_up).astype(np.float32) / 255.0

        hr = torch.from_numpy(hr).unsqueeze(0)
        lr = torch.from_numpy(lr).unsqueeze(0)

        return lr, hr

# ---------------- PSNR ----------------
def psnr(pred, target):
    pred = torch.clamp(pred, 0.0, 1.0)
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100
    return 10 * torch.log10(1 / mse)

# ---------------- TRAIN ----------------
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        batch_count = 0

        for lr, hr in pbar:
            lr, hr = lr.to(device), hr.to(device)

            residual = model(lr)
            sr = lr + residual
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss / batch_count:.4f}")

        scheduler.step()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss / len(train_loader):.4f}")
        validate(model, val_loader)

# ---------------- VALIDATE ----------------
def validate(model, loader):
    model.eval()
    avg_psnr = 0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            sr = lr + model(lr)
            avg_psnr += psnr(sr, hr).item()

    print(f"Validation PSNR: {avg_psnr / len(loader):.2f} dB")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    hr_train_dir = "C:\\Users\\Ritesh\\Documents\\new\\DIV2K_train_HR\\DIV2K_train_HR"
    hr_valid_dir = "C:\\Users\\Ritesh\\Documents\\new\\DIV2K_valid_HR\\DIV2K_valid_HR"

    batch_size = 16
    patch_size = 96
    scale = 2
    epochs = 30
    lr_rate = 1e-4

    train_dataset = DIV2KDataset(hr_train_dir, patch_size, scale)
    val_dataset = DIV2KDataset(hr_valid_dir, patch_size, scale)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs)

    torch.save(model.state_dict(), "srcnn.pth")
    print("Model saved as srcnn.pth")
