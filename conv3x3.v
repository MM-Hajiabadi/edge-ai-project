// conv3x3.v — ضرب‌کننده کانولوشن 3x3 (یک فیلتر، یک پیکسل خروجی)
// out = sum(pixel[i] * weight[i])  — همه مقادیر integer
// پورت‌ها signed [9:0] چون مقادیر zero-shifted می‌توانند از 8 بیت بیرون بزنند
module conv3x3 (
    input clk,
    input rst,
    input valid,
    input signed [9:0] px00, px01, px02,
    input signed [9:0] px10, px11, px12,
    input signed [9:0] px20, px21, px22,
    input signed [9:0] w00, w01, w02,
    input signed [9:0] w10, w11, w12,
    input signed [9:0] w20, w21, w22,
    output reg signed [31:0] acc
);
    // 9 ضرب‌کننده موازی (10 بیت × 10 بیت = 20 بیت)
    wire signed [19:0] p00 = px00 * w00;
    wire signed [19:0] p01 = px01 * w01;
    wire signed [19:0] p02 = px02 * w02;
    wire signed [19:0] p10 = px10 * w10;
    wire signed [19:0] p11 = px11 * w11;
    wire signed [19:0] p12 = px12 * w12;
    wire signed [19:0] p20 = px20 * w20;
    wire signed [19:0] p21 = px21 * w21;
    wire signed [19:0] p22 = px22 * w22;

    always @(posedge clk) begin
        if (rst)
            acc <= 32'd0;
        else if (valid)
            acc <= (32'(p00) + 32'(p01) + 32'(p02)) +
                   (32'(p10) + 32'(p11) + 32'(p12)) +
                   (32'(p20) + 32'(p21) + 32'(p22));
    end
endmodule
