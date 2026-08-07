import os, json
import numpy as np
import torch
from quant_inference import (load_weights, load_activation_ranges,
                              build_quantized_model, QMIN, QMAX)

DATA_DIR = os.path.join(os.path.dirname(__file__), "artifacts", "data_transformation", "processed")

def main():
    print("تولید بردار 16 فیلتر از یک پچ واقعی...")
    weights = load_weights()
    ranges = load_activation_ranges()
    layers = build_quantized_model(weights, ranges)
    enc1 = layers["enc1"]

    wz = enc1.w_int.numpy().astype(np.int64)[:, 0] - enc1.zp_w

    src_n = np.load(os.path.join(DATA_DIR, "src_test_normal.npy"))
    src_n = (src_n - src_n.min()) / (src_n.max() - src_n.min() + 1e-8)
    sample = torch.from_numpy(src_n[0].astype(np.float32))
    x_q = torch.clamp(torch.round(sample / enc1.scale_in) + enc1.zp_in, QMIN, QMAX)
    xz = (x_q - enc1.zp_in).numpy().astype(np.int64)

    row, col = 1, 1
    patch = xz[row:row+3, col:col+3].flatten().tolist()

    vectors = {
        "pixels": patch,
        "filters": []
    }
    for f in range(16):
        wflat = wz[f].flatten().tolist()
        expected = int(np.sum(np.array(patch) * np.array(wflat)))
        vectors["filters"].append({
            "f": f,
            "weights": wflat,
            "expected": expected
        })

    with open("test_vectors_16f.json", "w") as fp:
        json.dump(vectors, fp)
    print(f"✅ test_vectors_16f.json — 1 پچ × 16 فیلتر")

    print(f"  پچ: {patch[:5]}...")
    print(f"  فیلتر 0: expected={vectors['filters'][0]['expected']}")
    print(f"  فیلتر 15: expected={vectors['filters'][15]['expected']}")

if __name__ == "__main__":
    main()
