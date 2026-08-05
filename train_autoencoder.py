"""
آموزش Autoencoder — نسخه بهبودیافته
EPOCHS=100 + Early Stopping + LR Scheduler + Augmentation
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
SEED = 42
PATIENCE = 10

torch.manual_seed(SEED)
np.random.seed(SEED)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
    
    def forward(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.enc3(x))
        x = self.relu(self.dec1(x))
        x = self.relu(self.dec2(x))
        x = self.dec3(x)
        return x

def load_data():
    train = np.load(os.path.join(DATA_DIR, "train_normal.npy"))
    train = (train - train.min()) / (train.max() - train.min() + 1e-8)
    train = train.astype(np.float32)
    train_tensor = torch.from_numpy(train).unsqueeze(1)
    
    n = len(train_tensor)
    n_val = int(n * 0.1)
    val_tensor = train_tensor[:n_val]
    train_tensor = train_tensor[n_val:]
    
    print(f"  آموزش: {len(train_tensor)} نمونه")
    print(f"  اعتبارسنجی: {len(val_tensor)} نمونه")
    return train_tensor, val_tensor

def add_noise(x, noise_level=0.01):
    """Data Augmentation — نویز گاوسی سبک"""
    if np.random.rand() < 0.5:
        noise = torch.randn_like(x) * noise_level
        return x + noise
    return x

def train():
    print("=" * 60)
    print("آموزش بهبودیافته Autoencoder (100 دوره + Early Stopping)")
    print("=" * 60)
    
    train_tensor, val_tensor = load_data()
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=BATCH_SIZE)
    
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print(f"  پارامترها: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  شروع آموزش...")
    print()
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    start = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch[0]
            x = add_noise(x)  # Augmentation
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_tensor)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0]
                recon = model(x)
                loss = criterion(recon, x)
                val_loss += loss.item() * len(x)
        val_loss /= len(val_tensor)
        
        scheduler.step(val_loss)
        
        print(f"  دوره {epoch+1:3d}/{EPOCHS}: train={train_loss:.6f}, val={val_loss:.6f} (lr={optimizer.param_groups[0]['lr']:.2e})")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "autoencoder_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  ⏹️ Early Stopping در دوره {epoch+1} (بهترین: دوره {best_epoch})")
                break
    
    elapsed = time.time() - start
    print()
    print(f"✅ آموزش کامل شد! ({elapsed/60:.1f} دقیقه)")
    print(f"  بهترین val_loss: {best_val_loss:.6f} (دوره {best_epoch})")
    print("=" * 60)

if __name__ == "__main__":
    train()
