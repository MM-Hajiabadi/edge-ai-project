// conv3x3_16f.v — 16 فیلتر کانولوشن 3x3 موازی (ENC1)
// تولید خودکار — برای تست با Cocotb
module conv3x3_16f (
    input clk, input rst, input valid,
    // پیکسل‌های ورودی (مشترک بین همه فیلترها)
    input signed [9:0] px00, px01, px02,
    input signed [9:0] px10, px11, px12,
    input signed [9:0] px20, px21, px22,
    input signed [9:0] w0_00, w0_01, w0_02,
    input signed [9:0] w0_10, w0_11, w0_12,
    input signed [9:0] w0_20, w0_21, w0_22,
    output reg signed [31:0] acc0,
    input signed [9:0] w1_00, w1_01, w1_02,
    input signed [9:0] w1_10, w1_11, w1_12,
    input signed [9:0] w1_20, w1_21, w1_22,
    output reg signed [31:0] acc1,
    input signed [9:0] w2_00, w2_01, w2_02,
    input signed [9:0] w2_10, w2_11, w2_12,
    input signed [9:0] w2_20, w2_21, w2_22,
    output reg signed [31:0] acc2,
    input signed [9:0] w3_00, w3_01, w3_02,
    input signed [9:0] w3_10, w3_11, w3_12,
    input signed [9:0] w3_20, w3_21, w3_22,
    output reg signed [31:0] acc3,
    input signed [9:0] w4_00, w4_01, w4_02,
    input signed [9:0] w4_10, w4_11, w4_12,
    input signed [9:0] w4_20, w4_21, w4_22,
    output reg signed [31:0] acc4,
    input signed [9:0] w5_00, w5_01, w5_02,
    input signed [9:0] w5_10, w5_11, w5_12,
    input signed [9:0] w5_20, w5_21, w5_22,
    output reg signed [31:0] acc5,
    input signed [9:0] w6_00, w6_01, w6_02,
    input signed [9:0] w6_10, w6_11, w6_12,
    input signed [9:0] w6_20, w6_21, w6_22,
    output reg signed [31:0] acc6,
    input signed [9:0] w7_00, w7_01, w7_02,
    input signed [9:0] w7_10, w7_11, w7_12,
    input signed [9:0] w7_20, w7_21, w7_22,
    output reg signed [31:0] acc7,
    input signed [9:0] w8_00, w8_01, w8_02,
    input signed [9:0] w8_10, w8_11, w8_12,
    input signed [9:0] w8_20, w8_21, w8_22,
    output reg signed [31:0] acc8,
    input signed [9:0] w9_00, w9_01, w9_02,
    input signed [9:0] w9_10, w9_11, w9_12,
    input signed [9:0] w9_20, w9_21, w9_22,
    output reg signed [31:0] acc9,
    input signed [9:0] w10_00, w10_01, w10_02,
    input signed [9:0] w10_10, w10_11, w10_12,
    input signed [9:0] w10_20, w10_21, w10_22,
    output reg signed [31:0] acc10,
    input signed [9:0] w11_00, w11_01, w11_02,
    input signed [9:0] w11_10, w11_11, w11_12,
    input signed [9:0] w11_20, w11_21, w11_22,
    output reg signed [31:0] acc11,
    input signed [9:0] w12_00, w12_01, w12_02,
    input signed [9:0] w12_10, w12_11, w12_12,
    input signed [9:0] w12_20, w12_21, w12_22,
    output reg signed [31:0] acc12,
    input signed [9:0] w13_00, w13_01, w13_02,
    input signed [9:0] w13_10, w13_11, w13_12,
    input signed [9:0] w13_20, w13_21, w13_22,
    output reg signed [31:0] acc13,
    input signed [9:0] w14_00, w14_01, w14_02,
    input signed [9:0] w14_10, w14_11, w14_12,
    input signed [9:0] w14_20, w14_21, w14_22,
    output reg signed [31:0] acc14,
    input signed [9:0] w15_00, w15_01, w15_02,
    input signed [9:0] w15_10, w15_11, w15_12,
    input signed [9:0] w15_20, w15_21, w15_22,
    output reg signed [31:0] acc15
);

    always @(posedge clk) begin
        if (rst) begin
            acc0 <= 32'd0;
            acc1 <= 32'd0;
            acc2 <= 32'd0;
            acc3 <= 32'd0;
            acc4 <= 32'd0;
            acc5 <= 32'd0;
            acc6 <= 32'd0;
            acc7 <= 32'd0;
            acc8 <= 32'd0;
            acc9 <= 32'd0;
            acc10 <= 32'd0;
            acc11 <= 32'd0;
            acc12 <= 32'd0;
            acc13 <= 32'd0;
            acc14 <= 32'd0;
            acc15 <= 32'd0;
        end else if (valid) begin
            acc0 <= 32'(px00*w0_00)+32'(px01*w0_01)+32'(px02*w0_02)+
                     32'(px10*w0_10)+32'(px11*w0_11)+32'(px12*w0_12)+
                     32'(px20*w0_20)+32'(px21*w0_21)+32'(px22*w0_22);
            acc1 <= 32'(px00*w1_00)+32'(px01*w1_01)+32'(px02*w1_02)+
                     32'(px10*w1_10)+32'(px11*w1_11)+32'(px12*w1_12)+
                     32'(px20*w1_20)+32'(px21*w1_21)+32'(px22*w1_22);
            acc2 <= 32'(px00*w2_00)+32'(px01*w2_01)+32'(px02*w2_02)+
                     32'(px10*w2_10)+32'(px11*w2_11)+32'(px12*w2_12)+
                     32'(px20*w2_20)+32'(px21*w2_21)+32'(px22*w2_22);
            acc3 <= 32'(px00*w3_00)+32'(px01*w3_01)+32'(px02*w3_02)+
                     32'(px10*w3_10)+32'(px11*w3_11)+32'(px12*w3_12)+
                     32'(px20*w3_20)+32'(px21*w3_21)+32'(px22*w3_22);
            acc4 <= 32'(px00*w4_00)+32'(px01*w4_01)+32'(px02*w4_02)+
                     32'(px10*w4_10)+32'(px11*w4_11)+32'(px12*w4_12)+
                     32'(px20*w4_20)+32'(px21*w4_21)+32'(px22*w4_22);
            acc5 <= 32'(px00*w5_00)+32'(px01*w5_01)+32'(px02*w5_02)+
                     32'(px10*w5_10)+32'(px11*w5_11)+32'(px12*w5_12)+
                     32'(px20*w5_20)+32'(px21*w5_21)+32'(px22*w5_22);
            acc6 <= 32'(px00*w6_00)+32'(px01*w6_01)+32'(px02*w6_02)+
                     32'(px10*w6_10)+32'(px11*w6_11)+32'(px12*w6_12)+
                     32'(px20*w6_20)+32'(px21*w6_21)+32'(px22*w6_22);
            acc7 <= 32'(px00*w7_00)+32'(px01*w7_01)+32'(px02*w7_02)+
                     32'(px10*w7_10)+32'(px11*w7_11)+32'(px12*w7_12)+
                     32'(px20*w7_20)+32'(px21*w7_21)+32'(px22*w7_22);
            acc8 <= 32'(px00*w8_00)+32'(px01*w8_01)+32'(px02*w8_02)+
                     32'(px10*w8_10)+32'(px11*w8_11)+32'(px12*w8_12)+
                     32'(px20*w8_20)+32'(px21*w8_21)+32'(px22*w8_22);
            acc9 <= 32'(px00*w9_00)+32'(px01*w9_01)+32'(px02*w9_02)+
                     32'(px10*w9_10)+32'(px11*w9_11)+32'(px12*w9_12)+
                     32'(px20*w9_20)+32'(px21*w9_21)+32'(px22*w9_22);
            acc10 <= 32'(px00*w10_00)+32'(px01*w10_01)+32'(px02*w10_02)+
                     32'(px10*w10_10)+32'(px11*w10_11)+32'(px12*w10_12)+
                     32'(px20*w10_20)+32'(px21*w10_21)+32'(px22*w10_22);
            acc11 <= 32'(px00*w11_00)+32'(px01*w11_01)+32'(px02*w11_02)+
                     32'(px10*w11_10)+32'(px11*w11_11)+32'(px12*w11_12)+
                     32'(px20*w11_20)+32'(px21*w11_21)+32'(px22*w11_22);
            acc12 <= 32'(px00*w12_00)+32'(px01*w12_01)+32'(px02*w12_02)+
                     32'(px10*w12_10)+32'(px11*w12_11)+32'(px12*w12_12)+
                     32'(px20*w12_20)+32'(px21*w12_21)+32'(px22*w12_22);
            acc13 <= 32'(px00*w13_00)+32'(px01*w13_01)+32'(px02*w13_02)+
                     32'(px10*w13_10)+32'(px11*w13_11)+32'(px12*w13_12)+
                     32'(px20*w13_20)+32'(px21*w13_21)+32'(px22*w13_22);
            acc14 <= 32'(px00*w14_00)+32'(px01*w14_01)+32'(px02*w14_02)+
                     32'(px10*w14_10)+32'(px11*w14_11)+32'(px12*w14_12)+
                     32'(px20*w14_20)+32'(px21*w14_21)+32'(px22*w14_22);
            acc15 <= 32'(px00*w15_00)+32'(px01*w15_01)+32'(px02*w15_02)+
                     32'(px10*w15_10)+32'(px11*w15_11)+32'(px12*w15_12)+
                     32'(px20*w15_20)+32'(px21*w15_21)+32'(px22*w15_22);
        end
    end
endmodule