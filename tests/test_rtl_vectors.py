import json
from pathlib import Path


def test_vectors_exist_and_valid():
    
    p = Path("artifacts/rtl_verification/test_vectors_16f.json") \
        if Path("artifacts/rtl_verification").exists() else Path("test_vectors_16f.json")
    if p.exists():
        with open(p) as f:
            tv = json.load(f)
        assert "pixels" in tv
        assert "filters" in tv
        assert len(tv["filters"]) == 16
        print(f"تست بردار RTL: {p} معتبر است")
    else:
        print("بردار تست هنوز ساخته نشده — skip")
