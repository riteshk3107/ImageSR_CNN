"""
Improved SRCNN training script addressing:
- Perceptual loss (combining MSE with VGG features)
- Data augmentation (flip, rotate)
- Weight decay for regularization
- Better initialization
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import glob
import random
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------- Residual SRCNN MODEL (5-layer CNN) ----------------
class ResidualSRCNN(nn.Module):
    """
    Enhanced SRCNN with residual learning:
    - Architecture: 5 convolutional layers, all stride 1, padding chosen to preserve spatial size.
    - Kernel sizes: [9, 5, 5, 5, 5].
    - Channel depth: 1 -> 64 -> 64 -> 32 -> 32 -> 1.
    - Receptive field: 25x25 pixels on the input (1 + (9-1) + 4*(5-1)).

    The network predicts a high-frequency residual R, which is added to the bicubic-upsampled input B:
        SR = B + R = B + f_theta(B)
    """

    def __init__(self):
        super(ResidualSRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize weights using He initialization for better convergence
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x is the bicubic-upsampled Y channel
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual = self.conv5(out)
        # Return only the residual; the caller is responsible for SR = bicubic + residual.
        return residual

# ---------------- PERCEPTUAL LOSS (VGG-based) ----------------
class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG features to capture high-frequency details.
    Combines MSE loss with feature-based loss for better texture recovery.
    """
    def __init__(self, feature_layer=5, mse_weight=0.5, perceptual_weight=0.5):
        super(PerceptualLoss, self).__init__()
        self.mse_weight = mse_weight
        self.perceptual_weight = perceptual_weight
        
        # Use VGG16 features (pretrained on ImageNet)
        try:
            # Try new API first (torchvision >= 0.13)
            try:
                from torchvision.models import vgg16, VGG16_Weights
                vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            except:
                # Fallback to old API (torchvision < 0.13)
                try:
                    vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features
                except:
                    # Last resort: try direct import
                    import torchvision.models as models
                    vgg = models.vgg16(pretrained=True).features
            
            self.feature_extractor = nn.Sequential(*list(vgg.children())[:feature_layer]).eval()
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        except Exception as e:
            # Fallback: use simple MSE if VGG not available
            print(f"Warning: VGG not available ({e}), using MSE only")
            self.feature_extractor = None
        
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target):
        # MSE loss (pixel-level)
        mse = self.mse_loss(pred, target)
        
        # Perceptual loss (feature-level)
        if self.feature_extractor is not None:
            # Expand grayscale to 3 channels for VGG
            pred_3ch = pred.repeat(1, 3, 1, 1)
            target_3ch = target.repeat(1, 3, 1, 1)
            
            pred_features = self.feature_extractor(pred_3ch)
            target_features = self.feature_extractor(target_3ch)
            perceptual = self.mse_loss(pred_features, target_features)
        else:
            perceptual = torch.tensor(0.0, device=pred.device)
        
        total_loss = self.mse_weight * mse + self.perceptual_weight * perceptual
        return total_loss

# ---------------- DATASET WITH AUGMENTATION ----------------
class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, patch_size=96, scale=2, augment=True):
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        self.patch_size = patch_size
        self.scale = scale
        self.augment = augment

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_paths[idx]).convert("YCbCr")
        y, _, _ = hr_img.split()

        hr_w, hr_h = y.size
        crop_size = self.patch_size * self.scale

        # Random crop
        x = random.randint(0, max(0, hr_w - crop_size))
        y_pos = random.randint(0, max(0, hr_h - crop_size))
        
        hr_crop = y.crop((x, y_pos, x + crop_size, y_pos + crop_size))
        
        # Data augmentation: random horizontal/vertical flip and rotation
        if self.augment:
            if random.random() > 0.5:
                hr_crop = hr_crop.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                hr_crop = hr_crop.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() > 0.5:
                hr_crop = hr_crop.rotate(90)
        
        # Generate LR: downscale then upscale (matches test degradation)
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
    best_val_psnr = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        batch_count = 0

        for lr, hr in pbar:
            lr, hr = lr.to(device), hr.to(device)

            # Network predicts residual; super-resolved output is bicubic + residual.
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

        # Validation
        val_psnr = validate(model, val_loader)
        
        # Update learning rate scheduler (ReduceLROnPlateau needs validation metric)
        scheduler.step(val_psnr)
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss / len(train_loader):.4f}")
        
        # Save best model
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save(model.state_dict(), "srcnn_best.pth")
            print(f"Saved best model (PSNR: {best_val_psnr:.2f} dB)")

# ---------------- VALIDATE ----------------
def validate(model, loader):
    model.eval()
    avg_psnr = 0
    with torch.no_grad():
        for lr, hr in loader:
            lr, hr = lr.to(device), hr.to(device)
            residual = model(lr)
            sr = lr + residual
            avg_psnr += psnr(sr, hr).item()

    val_psnr = avg_psnr / len(loader)
    print(f"Validation PSNR: {val_psnr:.2f} dB")
    return val_psnr

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Directories are specified as relative paths for Kaggle compatibility.
    # Point these to the root folders containing DIV2K HR PNGs.
    hr_train_dir = "DIV2K_train_HR"
    hr_valid_dir = "DIV2K_valid_HR"

    batch_size = 16
    patch_size = 96
    scale = 2
    epochs = 30
    lr_rate = 1e-4
    weight_decay = 1e-4  # L2 regularization
    
    # Use augmentation for training, not for validation
    train_dataset = DIV2KDataset(hr_train_dir, patch_size, scale, augment=True)
    val_dataset = DIV2KDataset(hr_valid_dir, patch_size, scale, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = ResidualSRCNN().to(device)
    
    # Improved loss: Combine MSE with perceptual loss
    # Option 1: Perceptual loss (requires VGG)
    try:
        criterion = PerceptualLoss(mse_weight=0.7, perceptual_weight=0.3).to(device)
        print("Using Perceptual Loss (MSE + VGG features)")
    except:
        # Option 2: Pure MSE (fallback)
        criterion = nn.MSELoss()
        print("Using MSE Loss only")
    
    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler: reduce LR when validation plateaus
    # Note: ReduceLROnPlateau doesn't have 'verbose' in older PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Alternative: StepLR scheduler (uncomment if you prefer fixed schedule)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs)

    torch.save(model.state_dict(), "srcnn.pth")
    print("Model saved as srcnn.pth")
