"""
ساخت Dataset از MIMII DUE — تبدیل صدا به log-mel spectrogram
ذخیره در فایل npy برای آموزش سریع
"""
import os
import glob
import numpy as np
import librosa
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "mimii_due")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

# پارامترهای استخراج ویژگی (سبک برای سخت‌افزار بعدی)
SR = 16000
N_FFT = 512
HOP_LENGTH = 128
N_MELS = 32
MAX_FRAMES = 256  # ~2 ثانیه از هر فایل (با hop=128 و sr=16k)

def extract_log_mel(filepath):
    """استخراج log-mel spectrogram از یک فایل صوتی"""
    audio, _ = librosa.load(filepath, sr=SR, mono=True)
    # نرمال‌سازی
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    # برش به MAX_FRAMES فریم (~2 ثانیه)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # محدود کردن به MAX_FRAMES
    if log_mel.shape[1] > MAX_FRAMES:
        log_mel = log_mel[:, :MAX_FRAMES]
    return log_mel

def build_split(split_name, label):
    """ساخت ویژگی‌ها برای یک بخش دیتاست"""
    split_dir = os.path.join(DATA_DIR, split_name)
    files = glob.glob(os.path.join(split_dir, "*.wav"))
    files = [f for f in files if label in os.path.basename(f)]
    
    print(f"  پردازش {split_name} [{label}]: {len(files)} فایل")
    features = []
    for f in tqdm(files):
        try:
            feat = extract_log_mel(f)
            features.append(feat)
        except Exception as e:
            print(f"  ❌ خطا در {os.path.basename(f)}: {e}")
    
    return np.array(features)  # [N, 32, MAX_FRAMES]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("ساخت Dataset از MIMII DUE")
    print("=" * 60)
    
    # Train: فقط سالم
    train_normal = build_split("train", "normal")
    np.save(os.path.join(OUT_DIR, "train_normal.npy"), train_normal)
    print(f"  ✅ train_normal: {train_normal.shape}")
    
    # Source test: سالم و معیوب
    src_normal = build_split("source_test", "normal")
    src_anomaly = build_split("source_test", "anomaly")
    np.save(os.path.join(OUT_DIR, "src_test_normal.npy"), src_normal)
    np.save(os.path.join(OUT_DIR, "src_test_anomaly.npy"), src_anomaly)
    print(f"  ✅ src_test_normal: {src_normal.shape}")
    print(f"  ✅ src_test_anomaly: {src_anomaly.shape}")
    
    # Target test: سالم و معیوب
    tgt_normal = build_split("target_test", "normal")
    tgt_anomaly = build_split("target_test", "anomaly")
    np.save(os.path.join(OUT_DIR, "tgt_test_normal.npy"), tgt_normal)
    np.save(os.path.join(OUT_DIR, "tgt_test_anomaly.npy"), tgt_anomaly)
    print(f"  ✅ tgt_test_normal: {tgt_normal.shape}")
    print(f"  ✅ tgt_test_anomaly: {tgt_anomaly.shape}")
    
    print("=" * 60)
    print("✅ Dataset ساخته شد! فایل‌ها در data/processed/")
    print("=" * 60)

if __name__ == "__main__":
    main()
