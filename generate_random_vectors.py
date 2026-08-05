"""
تولید بردارهای تصادفی برای تست all-16-filters
- از داده واقعی، چندین موقعیت + هر 16 فیلتر
- خروجی: random_vectors.json
"""
import os, json
import numpy as np
import torch
from quant_inference import (load_weights, load_activation_ranges,
                              build_quantized_model, QMIN, QMAX)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")

def main():
    print("تولید بردارهای تصادفی (16 فیلتر × چند موقعیت)...")
    weights = load_weights()
    ranges = load_activation_ranges()
    layers = build_quantized_model(weights, ranges)
    enc1 = layers["enc1"]

    # وزن‌های zero-shifted همه 16 فیلتر: [16,1,3,3]
    wz = enc1.w_int.numpy().astype(np.int64)[:, 0] - enc1.zp_w  # [16,3,3]

    # داده واقعی
    src_n = np.load(os.path.join(DATA_DIR, "src_test_normal.npy"))
    src_n = (src_n - src_n.min()) / (src_n.max() - src_n.min() + 1e-8)
    sample = torch.from_numpy(src_n[0].astype(np.float32))
    x_q = torch.clamp(torch.round(sample / enc1.scale_in) + enc1.zp_in, QMIN, QMAX)
    xz = (x_q - enc1.zp_in).numpy().astype(np.int64)  # [32,256]

    vectors = []
    # چند موقعیت معتبر (ورودی 32x256, kernel 3x3 → خروجی 16x128, موقعیت‌ها 0..13, 0..125)
    positions = [(0,0), (1,1), (5,10), (13,125), (2,50), (7,0), (0,100)]
    for (row, col) in positions:
        patch = xz[row:row+3, col:col+3]  # [3,3]
        for f in range(16):
            expected = int(np.sum(patch * wz[f]))
            vectors.append({
                "f": f, "row": row, "col": col,
                "pixels": patch.flatten().tolist(),
                "weights": wz[f].flatten().tolist(),
                "expected": expected
            })

    with open("random_vectors.json", "w") as f:
        json.dump(vectors, f)
    print(f"✅ {len(vectors)} بردار تولید شد → random_vectors.json")

if __name__ == "__main__":
    main()
