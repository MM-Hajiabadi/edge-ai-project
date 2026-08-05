"""
ارزیابی بهبودیافته — ۳ معیار خطای بازسازی + نرمال‌سازی
"""
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_autoencoder import Autoencoder

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "autoencoder_best.pth")

def load_split(name):
    data = np.load(os.path.join(DATA_DIR, f"{name}.npy"))
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    return torch.from_numpy(data.astype(np.float32)).unsqueeze(1)

def compute_errors(model, data_loader):
    """محاسبه ۳ معیار خطا: MSE، MAE، MaxAbs"""
    model.eval()
    mse_list, mae_list, max_list = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            recon = model(batch)
            diff = (recon - batch).abs()
            mse_list.extend((diff ** 2).mean(dim=[1, 2, 3]).cpu().numpy())
            mae_list.extend(diff.mean(dim=[1, 2, 3]).cpu().numpy())
            max_list.extend(diff.amax(dim=[1, 2, 3]).cpu().numpy())
    return np.array(mse_list), np.array(mae_list), np.array(max_list)

def evaluate():
    print("=" * 60)
    print("ارزیابی بهبودیافته — ۳ معیار خطا")
    print("=" * 60)
    
    model = Autoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    
    # بارگذاری
    src_normal = load_split("src_test_normal")
    src_anomaly = load_split("src_test_anomaly")
    tgt_normal = load_split("tgt_test_normal")
    tgt_anomaly = load_split("tgt_test_anomaly")
    
    loader = torch.utils.data.DataLoader
    mse_sn, mae_sn, max_sn = compute_errors(model, loader(src_normal, batch_size=32))
    mse_sa, mae_sa, max_sa = compute_errors(model, loader(src_anomaly, batch_size=32))
    mse_tn, mae_tn, max_tn = compute_errors(model, loader(tgt_normal, batch_size=32))
    mse_ta, mae_ta, max_ta = compute_errors(model, loader(tgt_anomaly, batch_size=32))
    
    # محاسبه AUC برای هر معیار
    results = {}
    for name, err_n, err_a in [
        ("MSE", mse_sn, mse_sa),
        ("MAE", mae_sn, mae_sa),
        ("MaxAbs", max_sn, max_sa),
    ]:
        y_true = np.concatenate([np.zeros(len(err_n)), np.ones(len(err_a))])
        y_score = np.concatenate([err_n, err_a])
        results[f"source_{name}"] = roc_auc_score(y_true, y_score)
    
    for name, err_n, err_a in [
        ("MSE", mse_tn, mse_ta),
        ("MAE", mae_tn, mae_ta),
        ("MaxAbs", max_tn, max_ta),
    ]:
        y_true = np.concatenate([np.zeros(len(err_n)), np.ones(len(err_a))])
        y_score = np.concatenate([err_n, err_a])
        results[f"target_{name}"] = roc_auc_score(y_true, y_score)
    
    print("\nمعیار AUC بر اساس نوع خطا:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    
    # بهترین معیار
    best_source = max(results["source_MSE"], results["source_MAE"], results["source_MaxAbs"])
    best_target = max(results["target_MSE"], results["target_MAE"], results["target_MaxAbs"])
    
    print(f"\n  بهترین Source: {best_source:.4f}")
    print(f"  بهترین Target: {best_target:.4f}")

if __name__ == "__main__":
    evaluate()
