import os
from pathlib import Path
import numpy as np
import librosa
from tqdm import tqdm
from src.edge_ai_anomaly_detection.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def extract_log_mel(self, filepath: Path) -> np.ndarray:
       
        audio, _ = librosa.load(filepath, sr=self.config.sample_rate, mono=True)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        if log_mel.shape[1] > self.config.max_frames:
            log_mel = log_mel[:, :self.config.max_frames]
        return log_mel

    def process_split(self, split_name: str, label: str, output_name: str):
        
        split_dir = self.config.data_dir_root / split_name
        files = [f for f in split_dir.glob("*.wav") if label in f.name]
        features = []
        for f in tqdm(files, desc=f"{split_name}/{label}"):
            features.append(self.extract_log_mel(f))
        arr = np.array(features)
        out_path = self.config.processed_dir / f"{output_name}.npy"
        np.save(out_path, arr)
        print(f"✅ {output_name}: {arr.shape} → {out_path}")
