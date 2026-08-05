"""
EDA — اکتشاف داده MIMII DUE (پمپ صنعتی)
ساختار: فایل‌ها در پوشه‌های source_test / target_test / train
نوع (normal/anomaly) از نام فایل تشخیص داده می‌شود
"""
import os
import glob
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "mimii_due")

def count_files():
    """شمارش فایل‌ها بر اساس نام فایل"""
    print("=" * 60)
    print("1. شمارش فایل‌ها (بر اساس نام):")
    print("=" * 60)
    for split in ["train", "source_test", "target_test"]:
        split_dir = os.path.join(DATA_DIR, split)
        if not os.path.isdir(split_dir):
            print(f"  [{split}] ❌ پوشه وجود ندارد")
            continue
        all_files = glob.glob(os.path.join(split_dir, "*.wav"))
        normal = [f for f in all_files if "normal" in os.path.basename(f)]
        anomaly = [f for f in all_files if "anomaly" in os.path.basename(f)]
        print(f"  [{split}]")
        print(f"    سالم (normal): {len(normal)}")
        print(f"    معیوب (anomaly): {len(anomaly)}")
        print(f"    جمع: {len(all_files)}")
    print()

def analyze_sample(filepath, title):
    """تحلیل یک نمونه صوتی"""
    print(f"  --- {title} ---")
    audio, sr = librosa.load(filepath, sr=None)
    duration = len(audio) / sr
    print(f"  نام فایل: {os.path.basename(filepath)}")
    print(f"  نرخ نمونه‌برداری: {sr} Hz")
    print(f"  مدت: {duration:.2f} ثانیه")
    print(f"  تعداد نمونه: {len(audio)}")
    print(f"  دامنه: [{audio.min():.3f}, {audio.max():.3f}]")
    print(f"  RMS: {np.sqrt(np.mean(audio**2)):.4f}")
    print()
    return audio, sr

def plot_waveform_spectrogram(audio, sr, title, savepath):
    """رسم waveform + spectrogram"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    time = np.linspace(0, len(audio)/sr, len(audio))
    axes[0].plot(time, audio, linewidth=0.3)
    axes[0].set_title(f"Waveform — {title}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=512, hop_length=128, n_mels=32)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    img = axes[1].imshow(log_mel, aspect='auto', origin='lower',
                         extent=[0, len(audio)/sr, 0, 32])
    axes[1].set_title(f"Log-Mel Spectrogram — {title}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Mel bins")
    plt.colorbar(img, ax=axes[1], fraction=0.02)
    
    plt.tight_layout()
    plt.savefig(savepath, dpi=100)
    plt.close()
    print(f"  📁 ذخیره شد: {savepath}")

def main():
    count_files()
    
    # پیدا کردن نمونه‌ها از train (آموزش) و source_test (تست)
    train_dir = os.path.join(DATA_DIR, "train")
    source_dir = os.path.join(DATA_DIR, "source_test")
    
    train_files = glob.glob(os.path.join(train_dir, "*.wav")) if os.path.isdir(train_dir) else []
    source_files = glob.glob(os.path.join(source_dir, "*.wav")) if os.path.isdir(source_dir) else []
    
    train_normal = [f for f in train_files if "normal" in os.path.basename(f)]
    source_anomaly = [f for f in source_files if "anomaly" in os.path.basename(f)]
    
    if not train_normal or not source_anomaly:
        print("❌ فایل کافی پیدا نشد! مسیرها و نام فایل‌ها را بررسی کنید.")
        return
    
# تحلیل نمونه سالم (از train)
    n_audio, n_sr = analyze_sample(train_normal[0], "نمونه سالم (train)")
    plot_waveform_spectrogram(n_audio, n_sr, "Normal",
                              os.path.join(os.path.dirname(__file__), "eda_normal.png"))
    
    # تحلیل نمونه معیوب (از source_test)
    a_audio, a_sr = analyze_sample(source_anomaly[0], "نمونه معیوب (source_test)")
    plot_waveform_spectrogram(a_audio, a_sr, "Anomaly",
                              os.path.join(os.path.dirname(__file__), "eda_anomaly.png"))
    
    print("=" * 60)
    print("✅ EDA کامل شد! فایل‌های eda_normal.png و eda_anomaly.png ساخته شدند.")
    print("=" * 60)

if __name__ == "__main__":
    main()
