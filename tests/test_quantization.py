import torch
from src.edge_ai_anomaly_detection.components.quantization import quantize_tensor

def test_quantize_tensor_shape():

    t = torch.rand(10, 10) * 2 - 1
    q, scale, zp = quantize_tensor(t)
    assert q.shape == t.shape
    assert q.dtype == torch.int8
    assert scale > 0


def test_quantize_dequantize_approx():

    t = torch.randn(1000) * 0.5
    q, scale, zp = quantize_tensor(t)
    t_deq = (q.float() - zp) * scale
    err = (t - t_deq).abs()
    err_99 = torch.quantile(err, 0.99).item()
    assert err_99 < scale * 1.5, f"خطای 99% داده: {err_99}"
    print(f"max_err={err.max().item():.4f}, err@99%={err_99:.4f}, scale={scale:.4f}")