import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import json

pixel_names = ["px00","px01","px02","px10","px11","px12","px20","px21","px22"]

def signed32(val):
    
    val = int(val)
    if val >= 2**31:    
        val -= 2**32
    return val

@cocotb.test()
async def test_16_filters(dut):
    with open("test_vectors_16f.json") as f:
        tv = json.load(f)

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    for n, v in zip(pixel_names, tv["pixels"]):
        getattr(dut, n).value = v

    expected = {}
    for fdata in tv["filters"]:
        f = fdata["f"]
        w = fdata["weights"]

        for idx in range(9):
            r = idx // 3
            c = idx % 3
            wname = f"w{f}_{r}{c}"
            getattr(dut, wname).value = w[idx]
        expected[f] = fdata["expected"]

    dut.valid.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    ok = True
    for f in range(16):
        actual_raw = int(getattr(dut, f"acc{f}").value)
        actual = signed32(actual_raw)  
        exp = expected[f]
        status = "✅" if actual == exp else "❌"
        if actual != exp:
            ok = False
        print(f"  فیلتر {f:2d}: hw={actual}, ref={exp} {status}")

    assert ok, "برخی فیلترها ناهماهنگ هستند!"
    print("✅ همه 16 فیلتر بیت‌به‌بیت با مرجع برابرند!")
