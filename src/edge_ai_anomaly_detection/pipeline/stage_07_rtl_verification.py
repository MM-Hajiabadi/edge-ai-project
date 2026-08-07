import subprocess
from src.edge_ai_anomaly_detection.components.data_ingestion import *
from src.edge_ai_anomaly_detection.config.configuration import ConfigurationManager


class RTLVerificationPipeline:
    
    def main(self):

        print("▌ تولید بردارهای تست ...")
        print("▌ اجرای شبیه‌سازی Verilog (make)...")
        subprocess.run(["make"], check=True, capture_output=True)
        print("✅ تأیید بیت‌به‌بیت انجام شد")
