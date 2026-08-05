"""
تولید بردارهای تست برای ماژول Verilog (لایه enc1)
- از شبیه‌ساز کوانتیزه (مرجع طلایی) روی داده واقعی
- خروجی: test_vectors.json — ورودی‌های صحیح + خروجی مورد انتظار
"""
import os
import json
import numpy as np
import torch
from quant_inference import (load_weights, load_activation_ranges,
                              build_quantized_model, QMIN, QMAX)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

def main():
    print("تولید بردارهای تست برای conv3x3 (enc1, فیلتر 0)...")

    weights = load_weights()
    ranges = load_activation_ranges()
    layers = build_quantized_model(weights, ranges)
    enc1 = layers["enc1"]

    # نمونه واقعی اول از source_test_normal
    src_n = np.load(os.path.join(DATA_DIR, "src_test_normal.npy"))
    src_n = (src_n - src_n.min()) / (src_n.max() - src_n.min() + 1e-8)
    sample = torch.from_numpy(src_n[0].astype(np.float32))  # [32, 256]

    # کوانتیزه کردن ورودی با scale/zp لایه enc1
    x_q = torch.clamp(torch.round(sample / enc1.scale_in) + enc1.zp_in, QMIN, QMAX)
    x_zeroed = (x_q - enc1.zp_in).numpy().astype(np.int64)  # مقادیر صحیح

    # وزن فیلتر 0: w_int - zp_w
    w_int = enc1.w_int.numpy().astype(np.int64)  # [16,1,3,3]
    wz0 = (w_int[0, 0] - enc1.zp_w)              # [3,3]

    # یک پنجره 3x3 واقعی از ورودی (موقعیت (1,1))
    row, col = 1, 1
    patch = x_zeroed[row:row+3, col:col+3]

    # خروجی مورد انتظار (ضرب داخلی صحیح)
    expected = int(np.sum(patch * wz0))

    vectors = {
        "pixels": patch.flatten().tolist(),   # 9 ورودی (x_q - zp_in)
        "weights": wz0.flatten().tolist(),    # 9 وزن (w_int - zp_w)
        "expected": expected,
        "note": "conv3x3, enc1 filter0, position (1,1), no bias"
    }

    with open("test_vectors.json", "w") as f:
        json.dump(vectors, f, indent=2)

    print(f"پچ ورودی:  {patch.flatten().tolist()}")
    print(f"وزن‌ها:    {wz0.flatten().tolist()}")
    print(f"خروجی:     {expected}")
    print("✅ test_vectors.json ذخیره شد")

if __name__ == "__main__":
    main()
