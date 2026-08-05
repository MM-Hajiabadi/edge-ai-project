"""
کالیبراسیون Activations — پیدا کردن محدوده واقعی خروجی هر لایه
با عبور 100 نمونه سالم واقعی از مدل Float32
خروجی: activation_ranges.json (برای کوانتیزاسیون کامل)
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from train_autoencoder import Autoencoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder_best.pth")

def load_calibration_data(n_samples=100):
    """بارگذاری 100 نمونه سالم برای کالیبراسیون"""
    train = np.load(os.path.join(DATA_DIR, "train_normal.npy"))
    train = (train - train.min()) / (train.max() - train.min() + 1e-8)
    return torch.from_numpy(train[:n_samples].astype(np.float32)).unsqueeze(1)

def calibrate():
    print("=" * 60)
    print("کالیبراسیون Activations (محدوده واقعی میانی‌ها)")
    print("=" * 60)
    
    model = Autoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    calib_data = load_calibration_data(100)
    print(f"  داده کالیبراسیون: {calib_data.shape}")
    
    # ثبت min/max برای هر لایه
    layer_ranges = {}
    
    # تعریف قلاب‌ها (hooks) برای گرفتن خروجی لایه‌ها
    hooks = {}
    def make_hook(name):
        def hook(module, input, output):
            if name not in layer_ranges:
                layer_ranges[name] = {"min": float('inf'), "max": float('-inf')}
            layer_ranges[name]["min"] = min(layer_ranges[name]["min"], output.min().item())
            layer_ranges[name]["max"] = max(layer_ranges[name]["max"], output.max().item())
        return hook
    
    # ثبت hooks روی لایه‌های کانولوشن
    layer_names = ["enc1", "enc2", "enc3", "dec1", "dec2", "dec3"]
    hooks_handles = []
    for name in layer_names:
        layer = getattr(model, name)
        hooks_handles.append(layer.register_forward_hook(make_hook(name)))
    
    # عبور داده کالیبراسیون
    print("  عبور 100 نمونه از مدل...")
    with torch.no_grad():
        for i in range(0, len(calib_data), 10):
            batch = calib_data[i:i+10]
            model(batch)
    
    # حذف hooks
    for h in hooks_handles:
        h.remove()
    
    # چاپ نتایج
    print(f"\n  محدوده Activations (min/max):")
    for name in layer_names:
        r = layer_ranges[name]
        print(f"  {name}: min={r['min']:.6f}, max={r['max']:.6f}")
    
    # ذخیره
    with open(os.path.join(MODEL_DIR, "activation_ranges.json"), "w") as f:
        json.dump(layer_ranges, f, indent=2)
    print(f"\n✅ activation_ranges.json ذخیره شد")
    print("=" * 60)

if __name__ == "__main__":
    calibrate()
