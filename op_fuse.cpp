#include "Halide.h"
#include "common.h"

#include <stdio.h>

using namespace Halide;
using namespace Halide::Tools;

int main(int argc, char **argv) {
    const int N = 5, CI = 128, CO = 128, W = 100, H = 80, KW = 3, KH = 3;
    const int dilation = 31;
    const float epsilon = 1.e-9f;

    ImageParam input(type_of<float>(), 4);
    ImageParam filter(type_of<float>(), 4);

    // define a fused operator
    // TODO: you may rewrite this part
    Var x("x"), y("y"), c("c"), n("n");

    Func dilated_conv("dilated_conv");
    RDom r(0, CI, 0, KW, 0, KH);

    dilated_conv(c, x, y, n) = 0.0f;
    dilated_conv(c, x, y, n) += filter(c, r.y, r.z, r.x) * input(r.x, x + r.y * (dilation + 1), y + r.z * (dilation + 1), n);

    Func mu("mu"), sigma("sigma"), out("out"), tmp("tmp");
    Func inv_sqrt("inv_sqrt");
    RDom s(0, W, 0, H, 0, N);
    // Func sum_conv("sum_conv");
    // Var ci("ci"), kw("kw"), kh("kh");

    // sum_conv(ci, kw, kh) = Halide::sum(input(ci, s.x + kw * (dilation + 1), s.y + kh * (dilation + 1), s.z));
    // mu(c) = Halide::sum(filter(c, r.y, r.z, r.x) * sum_conv(r.x, r.y, r.z)) / (N * H * W);

    tmp(c, x, y, n) = dilated_conv(c, x, y, n);
    mu(c) = Halide::sum(tmp(c, s.x, s.y, s.z)) / (N * H * W);

    sigma(c) = Halide::sum(pow((tmp(c, s.x, s.y, s.z) - mu(c)), 2)) / (N * H * W);
    inv_sqrt(c) = 1 / sqrt(sigma(c) + epsilon);

    out(c, x, y, n) = (tmp(c, x, y, n) - mu(c)) * inv_sqrt(c);
    
    // TODO: write Halide schedules below
    Target target = get_jit_target_from_environment();
    const int vec = target.natural_vector_size<float>();

    const int tile_w = 4;
    const int tile_h = 4;
    Var co("co"), ci("ci"), xo("xo"), xi("xi"), yo("yo"), yi("yi"), t("t");

    out.split(c, co, ci, vec * tile_w)
        .split(x, xo, xi, tile_h)
        .reorder(ci, xi, xo, y, n, co)
        .vectorize(ci, vec)
        .unroll(ci)
        .unroll(xi)
        .parallel(y)
        .parallel(n)
        .parallel(co);

    mu.compute_at(out, co)
        .split(c, co, ci, vec * tile_w)
        .reorder(ci, co)
        .vectorize(ci, vec)
        .unroll(ci)
        .parallel(co);
    inv_sqrt.compute_at(out, co)
        .split(c, co, ci, vec * tile_w)
        .reorder(ci, co)
        .vectorize(ci, vec)
        .unroll(ci)
        .parallel(co);

    tmp.compute_at(out, co)
        .split(c, co, ci, vec * tile_w)
        .split(x, xo, xi, tile_h)
        .reorder(ci, xi, xo, y, n, co)
        .vectorize(ci, vec)
        .unroll(ci)
        .unroll(xi)
        .parallel(y)
        .parallel(n)
        .parallel(co);

    dilated_conv.compute_at(tmp, xo)
        .vectorize(c, vec)
        .unroll(c)
        .unroll(x)
        .unroll(y)
        .update()
        .reorder(c, x, y, r.x, r.y, r.z, n)
        .vectorize(c, vec)
        .unroll(c)
        .unroll(x)
        .unroll(y)
        .unroll(r.x, 2);

    filter.in().compute_at(dilated_conv, r.x)
        .vectorize(_0, vec)
        .unroll(_3);
    input.in().compute_at(dilated_conv, x)
        .unroll(_0);

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
    // call dilated conv and bnorm seperately in oneDNN
    double t_onednn = dnnl_dilated_conv_wrapper(in.data(), fil.data(), output_ref.data(), {N, CI, CO, W, H, KW, KH, dilation, dilation});
    t_onednn += dnnl_batch_normalization_wrapper(output_ref.data(), epsilon, {N, CO, H, W});

    // check results
    if (check_equal<float, 4>(output_ref, output_halide)) {
        printf("Halide results - OK\n");
    } else {
        printf("Halide results - FAIL\n");
        return 1;
    }

    printf("Halide: %fms\n", t_halide * 1e3);
    printf("oneDNN: %fms\n\n", t_onednn * 1e3);

    printf("Success!\n");

    return 0;
}
