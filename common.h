#ifndef WRAPPER_H
#define WRAPPER_H

#include "oneapi/dnnl/dnnl.hpp"
#include "example_utils.hpp"
#include "halide_benchmark.h"

using namespace dnnl;
using namespace Halide;
using namespace Halide::Tools;

using tag = memory::format_tag;
using dt = memory::data_type;

struct ConvConfig {
    int N, CI, CO, W, H, KW, KH, DW, DH;
};

struct BNConfig {
    int N, C, H, W;
};

template <typename T, int D>
inline void random_data(Buffer<T, D> &b) {
    b.for_each_value([](T &value) {
        value = (T)(rand() % 256) / 256.0f;
    });
}

template <typename T, int D>
class Checker {
 public:
    bool equal;
    const Buffer<T, D> &b1;
    const Buffer<T, D> &b2;
    Checker(const Buffer<T, D> &b1, const Buffer<T, D> &b2) : b1(b1), b2(b2) {
        equal = true;
    }
    template<typename... Args>
    void operator() (Args... args) {
       equal &= (std::abs(b1(args...) - b2(args...)) < 0.001f);
    }
};

template <typename T, int D>
inline bool check_equal(const Buffer<T, D> &b1, const Buffer<T, D> &b2) {
    Checker<T, D> checker = Checker<T, D>(b1, b2);
    b1.for_each_element(checker);
    return checker.equal;
}

// dnnl wrapper
inline double dnnl_dilated_conv_wrapper(float *src, float *weight, float *dst, ConvConfig c) {
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream(engine);

    memory::dims src_dims = {c.N, c.CI, c.H + (c.KH - 1) * (c.DH + 1), c.W + (c.KW - 1) * (c.DW + 1)};
    memory::dims weights_dims = {c.CO, c.CI, c.KH, c.KW};
    memory::dims dst_dims = {c.N, c.CO, c.H, c.W};
    memory::dims strides_dims = {1, 1};
    memory::dims dilates_dims = {c.DH, c.DW};
    memory::dims padding_dims_l = {0, 0};
    memory::dims padding_dims_r = {0, 0};

    // Create memory objects for tensor data (src, weights, dst).
    // NHWC layout is assumed for src and dst, and IHWO for weights.
    auto user_src_mem = memory({src_dims, dt::f32, tag::nhwc}, engine);
    auto user_weights_mem = memory({weights_dims, dt::f32, tag::ihwo}, engine);
    auto user_dst_mem = memory({dst_dims, dt::f32, tag::nhwc}, engine);

    // Create memory descriptors with format_tag::any for the primitive. This
    // enables the convolution primitive to choose memory layouts for an
    // optimized primitive implementation, and these layouts may differ from the
    // ones provided by the user.
    auto conv_src_md = memory::desc(src_dims, dt::f32, tag::any);
    auto conv_weights_md = memory::desc(weights_dims, dt::f32, tag::any);
    auto conv_dst_md = memory::desc(dst_dims, dt::f32, tag::any);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src, user_src_mem);
    write_to_dnnl_memory(weight, user_weights_mem);

    // Create operation descriptor.
    auto conv_desc = convolution_forward::desc(
        prop_kind::forward_training, algorithm::convolution_direct, 
        conv_src_md, conv_weights_md, conv_dst_md, strides_dims,
        dilates_dims, padding_dims_l, padding_dims_r);

    // Create primitive descriptor.
    auto conv_pd = convolution_forward::primitive_desc(conv_desc, engine);

    auto conv_src_mem = user_src_mem;
    auto conv_weights_mem = user_weights_mem;
    auto conv_dst_mem = user_dst_mem;

    // Reorder the data in case the src and weights memory layouts generated by
    // the primitive and the ones provided by the user are different. In this
    // case, we create additional memory objects with internal buffers that will
    // contain the reordered data. The data in dst will be reordered after the
    // convolution computation has finalized.
    if (conv_pd.src_desc() != user_src_mem.get_desc()) {
        conv_src_mem = memory(conv_pd.src_desc(), engine);
        reorder(user_src_mem, conv_src_mem)
            .execute(engine_stream, user_src_mem, conv_src_mem);
    }
    if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
        conv_weights_mem = memory(conv_pd.weights_desc(), engine);
        reorder(user_weights_mem, conv_weights_mem)
            .execute(engine_stream, user_weights_mem, conv_weights_mem);
    }
    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        conv_dst_mem = memory(conv_pd.dst_desc(), engine);
    }

    // Create the primitive.
    auto conv_prim = convolution_forward(conv_pd);
    // Primitive arguments.
    std::unordered_map<int, memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
    conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

    // Primitive execution
    double t = benchmark(10, 10, [&]() {
        conv_prim.execute(engine_stream, conv_args);
        engine_stream.wait();
    });

    // Reorder the data in case the dst memory descriptor generated by the
    // primitive and the one provided by the user are different.
    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        reorder(conv_dst_mem, user_dst_mem)
            .execute(engine_stream, conv_dst_mem, user_dst_mem);
    } else
        user_dst_mem = conv_dst_mem;
    read_from_dnnl_memory(dst, user_dst_mem);

    return t;
}

inline double dnnl_batch_normalization_wrapper(float *src, const float epsilon, BNConfig c) {
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    dnnl::stream engine_stream(engine);

    memory::dims src_dims = {c.N, c.C, c.H, c.W};

    auto src_md = memory::desc(src_dims, dt::f32, tag::nhwc);
    auto src_mem = memory(src_md, engine);

    // Write data to memory object's handle.
    write_to_dnnl_memory(src, src_mem);

    // Create operation descriptor.
    auto bnorm_d = batch_normalization_forward::desc(
        prop_kind::forward_training, src_md, epsilon,
        normalization_flags::none);

    // Create primitive descriptor.
    auto bnorm_pd = batch_normalization_forward::primitive_desc(bnorm_d, engine);

    auto mean_mem = memory(bnorm_pd.mean_desc(), engine);
    auto variance_mem = memory(bnorm_pd.variance_desc(), engine);

    // Create the primitive.
    auto bnorm_prim = batch_normalization_forward(bnorm_pd);
    // Primitive arguments. Set up in-place execution by assigning src as DST.
    std::unordered_map<int, memory> bnorm_args;
    bnorm_args.insert({DNNL_ARG_SRC, src_mem});
    bnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
    bnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
    bnorm_args.insert({DNNL_ARG_DST, src_mem});

    // Primitive execution
    double t = benchmark(10, 10, [&]() {
        bnorm_prim.execute(engine_stream, bnorm_args);
        engine_stream.wait();
    });

    // Read data from memory object's handle.
    read_from_dnnl_memory(src, src_mem);

    return t;
}

#endif
