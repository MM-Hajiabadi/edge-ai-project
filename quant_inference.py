"""
شبیه‌ساز کوانتیزه کامل (Integer-Only Inference Simulator)
نسخه 2: محاسبات integer-like (صفر کردن zero_point قبل از کانولوشن)
- کوانتیزه کردن ورودی با scale/zp خودش
- کانولوشن روی مقادیر zero-shifted (شبیه integer arithmetic)
- کوانتیزه کردن خروجی با scale/zp خودش
- Golden Reference برای Verilog
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

QMIN, QMAX = -128, 127  # INT8 range


def compute_scale_zp(t_min, t_max):
    """محاسبه scale و zero_point از محدوده (نامتقارن برای activations)"""
    scale = (t_max - t_min) / (QMAX - QMIN)
    if scale == 0:
        scale = 1e-8
    zp = QMIN - round(t_min / scale)
    zp = max(QMIN, min(QMAX, zp))
    return scale, zp


def load_weights():
    with open(os.path.join(MODEL_DIR, "golden_weights.json")) as f:
        return json.load(f)


def load_activation_ranges():
    with open(os.path.join(MODEL_DIR, "activation_ranges.json")) as f:
        return json.load(f)


class QuantizedLayer:
    """لایه کوانتیزه با محاسبات integer-like (شبیه سخت‌افزار واقعی)"""

    def __init__(self, w_int, shape, scale_w, zp_w, scale_in, zp_in, scale_out, zp_out):
        # وزن INT8 (در محدوده [-128, 127])
        self.w_int = torch.tensor(w_int).reshape(shape).float()
        self.scale_w = scale_w
        self.zp_w = zp_w
        self.scale_in = scale_in
        self.zp_in = zp_in
        self.scale_out = scale_out
        self.zp_out = zp_out

    def forward(self, x, is_transpose=False, stride=2, padding=1, output_padding=1):
        # 1. کوانتیزه کردن ورودی به INT8
        x_q = torch.clamp(torch.round(x / self.scale_in) + self.zp_in, QMIN, QMAX)

        # 2. صفر کردن zero_point ورودی: (x_q - zp_in)
        x_zeroed = x_q - self.zp_in

        # 3. صفر کردن zero_point وزن: (w_int - zp_w)
        w_zeroed = self.w_int - self.zp_w

        # 4. کانولوشن روی مقادیر zero-shifted (integer-like)
        if is_transpose:
            out_int = F.conv_transpose2d(x_zeroed, w_zeroed,
                                         stride=stride, padding=padding, output_padding=output_padding)
        else:
            out_int = F.conv2d(x_zeroed, w_zeroed, stride=stride, padding=padding)

        # 5. دیکوانتیزه: out_float = out_int * scale_in * scale_w
        out_float = out_int * (self.scale_in * self.scale_w)

        # 6. کوانتیزه کردن خروجی به مقیاس خروجی (برای لایه بعد)
        out_q = torch.clamp(torch.round(out_float / self.scale_out) + self.zp_out, QMIN, QMAX)

        # 7. برگرداندن به float (دیکوانتیزه) — دقت لایه بعد حفظ شود
        return (out_q - self.zp_out) * self.scale_out


def build_quantized_model(weights, ranges):
    """ساخت مدل کوانتیزه کامل"""
    layers = {}

    # محدوده ورودی مدل: log-mel نرمال‌شده در [0,1]
    in_min, in_max = 0.0, 1.0

    # enc1: ورودی = خود مدل (0..1)
    r_enc1 = ranges["enc1"]
    layers["enc1"] = QuantizedLayer(
        weights["enc1"]["int8_weights"], weights["enc1"]["shape"],
        weights["enc1"]["scale"], weights["enc1"]["zero_point"],
        *compute_scale_zp(in_min, in_max),
        *compute_scale_zp(r_enc1["min"], r_enc1["max"])
    )

    # enc2: ورودی = خروجی enc1
    r_enc2 = ranges["enc2"]
    layers["enc2"] = QuantizedLayer(
        weights["enc2"]["int8_weights"], weights["enc2"]["shape"],
        weights["enc2"]["scale"], weights["enc2"]["zero_point"],
        *compute_scale_zp(r_enc1["min"], r_enc1["max"]),
        *compute_scale_zp(r_enc2["min"], r_enc2["max"])
    )

    # enc3
    r_enc3 = ranges["enc3"]
    layers["enc3"] = QuantizedLayer(
        weights["enc3"]["int8_weights"], weights["enc3"]["shape"],
        weights["enc3"]["scale"], weights["enc3"]["zero_point"],
        *compute_scale_zp(r_enc2["min"], r_enc2["max"]),
        *compute_scale_zp(r_enc3["min"], r_enc3["max"])
    )

    # dec1
    r_dec1 = ranges["dec1"]
    layers["dec1"] = QuantizedLayer(
        weights["dec1"]["int8_weights"], weights["dec1"]["shape"],
        weights["dec1"]["scale"], weights["dec1"]["zero_point"],
        *compute_scale_zp(r_enc3["min"], r_enc3["max"]),
        *compute_scale_zp(r_dec1["min"], r_dec1["max"])
    )

    # dec2
    r_dec2 = ranges["dec2"]
    layers["dec2"] = QuantizedLayer(
        weights["dec2"]["int8_weights"], weights["dec2"]["shape"],
        weights["dec2"]["scale"], weights["dec2"]["zero_point"],
        *compute_scale_zp(r_dec1["min"], r_dec1["max"]),
        *compute_scale_zp(r_dec2["min"], r_dec2["max"])
    )

    # dec3 (خروجی نهایی)
    r_dec3 = ranges["dec3"]
    layers["dec3"] = QuantizedLayer(
        weights["dec3"]["int8_weights"], weights["dec3"]["shape"],
        weights["dec3"]["scale"], weights["dec3"]["zero_point"],
        *compute_scale_zp(r_dec2["min"], r_dec2["max"]),
        *compute_scale_zp(r_dec3["min"], r_dec3["max"])
    )

    return layers


def quant_inference(x, layers):
    """اجرای کامل مدل کوانتیزه"""
    # Encoder
    x = F.relu(layers["enc1"].forward(x))
    x = F.relu(layers["enc2"].forward(x))
    x = F.relu(layers["enc3"].forward(x))
    # Decoder
    x = F.relu(layers["dec1"].forward(x, is_transpose=True))
    x = F.relu(layers["dec2"].forward(x, is_transpose=True))
    x = layers["dec3"].forward(x, is_transpose=True)
    return x


def main():
    print("=" * 60)
    print("شبیه‌ساز کوانتیزه کامل v2 (integer-like)")
    print("=" * 60)

    weights = load_weights()
    ranges = load_activation_ranges()
    layers = build_quantized_model(weights, ranges)
    print("✅ مدل کوانتیزه کامل ساخته شد")

    # بارگذاری داده تست
    src_n = np.load(os.path.join(DATA_DIR, "src_test_normal.npy"))
    src_a = np.load(os.path.join(DATA_DIR, "src_test_anomaly.npy"))
    src_n = (src_n - src_n.min()) / (src_n.max() - src_n.min() + 1e-8)
    src_a = (src_a - src_a.min()) / (src_a.max() - src_a.min() + 1e-8)

    def compute_errors(data):
        errors = []
        for i in range(len(data)):
            s = torch.from_numpy(data[i].astype(np.float32)).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                recon = quant_inference(s, layers)
            errors.append(((recon - s) ** 2).mean().item())
        return np.array(errors)

    print("  محاسبه خطا روی source_test (300+300)...")
    err_n = compute_errors(src_n)
    err_a = compute_errors(src_a)

    y_true = np.concatenate([np.zeros(len(err_n)), np.ones(len(err_a))])
    y_score = np.concatenate([err_n, err_a])
    auc = roc_auc_score(y_true, y_score)

    print(f"\n  میانگین خطا — سالم: {err_n.mean():.6f}")
    print(f"  میانگین خطا — معیوب: {err_a.mean():.6f}")
    print(f"  AUC (کوانتیزه کامل v2): {auc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
