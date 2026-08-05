"""
تولیدکننده کد Verilog — ساخت conv3x3_16f.v با 16 فیلتر موازی
خروجی: فایل RTL کامل برای لایه enc1 (نسخه بدون خطای تکرار)
"""
def generate_module():
    lines = []
    lines.append("// conv3x3_16f.v — 16 فیلتر کانولوشن 3x3 موازی (ENC1)")
    lines.append("// تولید خودکار — برای تست با Cocotb")
    lines.append("module conv3x3_16f (")
    lines.append("    input clk, input rst, input valid,")
    lines.append("    // پیکسل‌های ورودی (مشترک بین همه فیلترها)")
    lines.append("    input signed [9:0] px00, px01, px02,")
    lines.append("    input signed [9:0] px10, px11, px12,")
    lines.append("    input signed [9:0] px20, px21, px22,")

    # پورت‌های وزن + خروجی برای هر 16 فیلتر — همه با ویرگول
    all_ports = []
    for f in range(16):
        all_ports.append(f"    input signed [9:0] w{f}_00, w{f}_01, w{f}_02,")
        all_ports.append(f"    input signed [9:0] w{f}_10, w{f}_11, w{f}_12,")
        all_ports.append(f"    input signed [9:0] w{f}_20, w{f}_21, w{f}_22,")
        all_ports.append(f"    output reg signed [31:0] acc{f},")

    # لازم: آخرین پورت هر خط خودش ویرگول دارد؛ ولی پورت آخر module نباید ویرگول داشته باشد.
    # ساده‌ترین: همه را با ویرگول بنویس، بعد آخرین را ویرایش کن.
    last_port = all_ports[-1]                   # آخرین: "output reg ... acc15,"
    all_ports[-1] = last_port[:-1]              # حذف ویرگول آخر → "output reg ... acc15"

    lines.extend(all_ports)
    lines.append(");")
    lines.append("")
    lines.append("    always @(posedge clk) begin")
    lines.append("        if (rst) begin")
    for f in range(16):
        lines.append(f"            acc{f} <= 32'd0;")
    lines.append("        end else if (valid) begin")

    for f in range(16):
        lines.append(f"            acc{f} <= 32'(px00*w{f}_00)+32'(px01*w{f}_01)+32'(px02*w{f}_02)+")
        lines.append(f"                     32'(px10*w{f}_10)+32'(px11*w{f}_11)+32'(px12*w{f}_12)+")
        lines.append(f"                     32'(px20*w{f}_20)+32'(px21*w{f}_21)+32'(px22*w{f}_22);")

    lines.append("        end")
    lines.append("    end")
    lines.append("endmodule")
    return "\n".join(lines)

def main():
    rtl = generate_module()
    with open("conv3x3_16f.v", "w") as f:
        f.write(rtl)
    print("✅ conv3x3_16f.v تولید شد (بدون تکرار):")
    # بررسی: چند بار w15 ظاهر شده (باید فقط یک بار در پورت‌ها)
    count_w15 = rtl.count("w15_20")
    print(f"  w15_20 اعلام شده: {count_w15} بار (باید 1 بار در پورت + استفاده در محاسبه)")
    print(f"  خطوط: {len(rtl.splitlines())}")

if __name__ == "__main__":
    main()
