import os
import time
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.edge_ai_anomaly_detection.entity.config_entity import ModelTrainerConfig


class Autoencoder(nn.Module):
    
    def __init__(self, channels=(16, 32, 64)):
        super().__init__()
        self.enc1 = nn.Conv2d(1, channels[0], kernel_size=3, stride=2, padding=1)
        self.enc2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.dec1 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec2 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec3 = nn.ConvTranspose2d(channels[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.enc3(x))
        x = self.relu(self.dec1(x))
        x = self.relu(self.dec2(x))
        x = self.dec3(x)
        return x


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig, data_dir: Path):
        self.config = config
        self.data_dir = data_dir 
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def load_data(self):
        
        train = np.load(self.data_dir / "train_normal.npy")
        train = (train - train.min()) / (train.max() - train.min() + 1e-8)
        train_tensor = torch.from_numpy(train.astype(np.float32)).unsqueeze(1)
        n = len(train_tensor)
        n_val = int(n * self.config.val_split)
        val_tensor = train_tensor[:n_val]
        train_tensor = train_tensor[n_val:]
        return train_tensor, val_tensor

    def train(self):
        print("=" * 50)
        print("آموزش Autoencoder (از config)")
        print("=" * 50)
        train_tensor, val_tensor = self.load_data()
        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_tensor), batch_size=self.config.batch_size)

        model = Autoencoder(self.config.channels if hasattr(self.config, 'channels') else (16, 32, 64))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        best_val = float('inf')
        patience_ctr = 0
        start = time.time()

        for epoch in range(self.config.epochs):
            model.train()
            tl = 0.0
            for batch in train_loader:
                x = batch[0]
                
                if np.random.rand() < 0.4:
                    x = x + torch.randn_like(x) * self.config.noise_augmentation
                optimizer.zero_grad()
                recon = model(x)
                loss = criterion(recon, x)
                loss.backward()
                optimizer.step()
                tl += loss.item() * len(x)
            tl /= len(train_tensor)

            model.eval()
            vl = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0]
                    recon = model(x)
                    vl += criterion(recon, x).item() * len(x)
            vl /= len(val_tensor)

            print(f"  دوره {epoch+1}/{self.config.epochs}: train={tl:.6f}, val={vl:.6f}")

            if vl < best_val:
                best_val = vl
                patience_ctr = 0
                self.config.model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), self.config.model_path)
            else:
                patience_ctr += 1
                if patience_ctr >= self.config.patience:
                    print(f"⏹️ Early stop at epoch {epoch+1}")
                    break

       
        log = {
            "best_val_loss": best_val,
            "epochs_run": epoch+1,
            "time_minutes": (time.time()-start)/60,
            "params": {"batch_size": self.config.batch_size, "epochs": self.config.epochs,
                       "lr": self.config.learning_rate}
        }
        self.config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.metrics_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"✅ مدل: {self.config.model_path}")
        print(f"✅ لاگ: {self.config.metrics_path}")
