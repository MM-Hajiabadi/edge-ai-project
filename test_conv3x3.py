import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import json

@cocotb.test()
async def test_conv3x3_enc1(dut):
    """مقایسه بیت‌به‌بیت: Verilog vs مرجع طلایی"""
    with open("test_vectors.json") as f:
        tv = json.load(f)

    pixels = tv["pixels"]    # 9 مقدار
    weights = tv["weights"]  # 9 مقدار
    expected = tv["expected"]

    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())

    # ریست
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    # ست کردن ورودی‌ها
    names_px = ["px00","px01","px02","px10","px11","px12","px20","px21","px22"]
    names_w  = ["w00","w01","w02","w10","w11","w12","w20","w21","w22"]
    for n, v in zip(names_px, pixels):
        getattr(dut, n).value = v
    for n, v in zip(names_w, weights):
        getattr(dut, n).value = v
    dut.valid.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    actual = int(dut.acc.value)
    print(f"خروجی Verilog:     {actual}")
    print(f"خروجی مرجع طلایی:  {expected}")
    assert actual == expected, f"FAIL: {actual} != {expected}"
    print("✅ تطابق بیت‌به‌بیت: سخت‌افزار == مرجع پایتونی")
