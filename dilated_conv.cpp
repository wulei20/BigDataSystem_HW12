#include "Halide.h"
#include "common.h"

#include <stdio.h>

using namespace Halide;
using namespace Halide::Tools;

int main(int argc, char **argv) {
    const int N = 5, CI = 128, CO = 128, W = 100, H = 80, KW = 3, KH = 3;
    const int dilation = 31;

    ImageParam input(type_of<float>(), 4);
    ImageParam filter(type_of<float>(), 4);

    // define dilated convolution
    // you can also rewrite algorithm definition part, as long as results are correct
    Var x("x"), y("y"), c("c"), n("n");

    Func dilated_conv("dilated_conv"), out("out");
    RDom r(0, CI, 0, KW, 0, KH);

    dilated_conv(c, x, y, n) = 0.0f;
    dilated_conv(c, x, y, n) += filter(c, r.y, r.z, r.x) * input(r.x, x + r.y * (dilation + 1), y + r.z * (dilation + 1), n);
    out(c, x, y, n) = dilated_conv(c, x, y, n);

    // TODO: write Halide schedules below
    // 获取目标设备向量宽度
    Target target = get_jit_target_from_environment();
    const int vec = target.natural_vector_size<float>();

    // 调整块大小
    const int tile_w = 4;
    const int tile_h = 4;

    Var co("co"), ci("ci"), xo("xo"), xi("xi"), yo("yo"), yi("yi"), t("t");

    // 主函数 out 的调度
    out.split(c, co, ci, vec * tile_w)
        .split(x, xo, xi, tile_h)
        .reorder(ci, xi, xo, y, n, co)
        .vectorize(ci, vec)        // 按自然向量宽度进行向量化
        .unroll(ci)               // 对小范围的 `ci` 展开
        .unroll(xi)               // 展开块内的 x
        .parallel(y)              // 对输出的 y 维度并行化
        .parallel(n)              // 对批次并行化
        .parallel(co);            // 并行处理通道块

    // 中间计算 dilated_conv 的调度
    dilated_conv.compute_at(out, xo)
        .vectorize(c, vec)        // 按 c 向量化
        .unroll(c)                // 对 c 展开
        .unroll(x)                // 展开 x 块
        .unroll(y)                // 展开 y 块
        .update()
        .reorder(c, x, y, r.x, r.y, r.z, n)
        .vectorize(c, vec)        // 归约计算的向量化
        .unroll(c)                // 对 c 展开
        .unroll(x)                // 对 x 展开
        .unroll(y)                // 对 y 展开
        .unroll(r.x, 2);          // 对 r.x 进行展开

    // 数据预处理的调度
    filter.in().compute_at(dilated_conv, r.x)
        .vectorize(_0, vec)       // 卷积核向量化
        .unroll(_0)               // 展开内部维度
        .unroll(_3);              // 展开通道维度

    input.in().compute_at(dilated_conv, x)
        .unroll(_0);              // 对通道展开

    

    Buffer<float, 4> in(CI, W + (KW - 1) * (dilation + 1), H + (KH - 1) * (dilation + 1), N);
    Buffer<float, 4> fil(CO, KW, KH, CI);
    Buffer<float, 4> output_halide(CO, W, H, N);

    // init randomly
    random_data<float, 4>(in);
    random_data<float, 4>(fil);
    input.set(in);
    filter.set(fil);

    // jit compile and warm-up
    out.realize(output_halide);
    // NOTE: uncomment next line if time is unstable
    // double t_halide = benchmark(10, 10, [&]() { dilated_conv.realize(output_halide); });
    double t_halide = benchmark(1, 1, [&]() { out.realize(output_halide); });

    Buffer<float, 4> output_ref(CO, W, H, N);
    // create and execute a dilated conv primitive using oneDNN
    double t_onednn = dnnl_dilated_conv_wrapper(in.data(), fil.data(), output_ref.data(), {N, CI, CO, W, H, KW, KH, dilation, dilation});

    // check results
    if (check_equal<float, 4>(output_ref, output_halide)) {
        printf("Halide results - OK\n");
    } else {
        printf("Halide results - FAIL\n");
        return 1;
    }

    float gflops = 2.0f * (N * CO * H * W) * (CI * KH * KW) / 1e9f;

    printf("Halide: %fms, %f GFLOP/s\n", t_halide * 1e3, (gflops / t_halide));
    printf("oneDNN: %fms, %f GFLOP/s\n\n", t_onednn * 1e3, (gflops / t_onednn));

    printf("Success!\n");

    return 0;
}
