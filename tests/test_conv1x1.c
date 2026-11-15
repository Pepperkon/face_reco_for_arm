// Compares conv_1x1_arm_neon (with mocked NEON) against conv_1x1_x86.
// Build from repo root:
//   gcc -O2 -fopenmp -DNEON_MOCK -I. tests/test_conv1x1.c -lm -o test_conv1x1
#define main engine_main
#include "../main.c"
#undef main

static float rand_float(void) {
    return (float) rand() / (float) RAND_MAX * 2.0f - 1.0f;
}

static int run_case(const uint32_t batch, const uint32_t c_in, const uint32_t c_out, const uint32_t h,
                    const uint32_t w, const uint32_t stride, const bool relu, const bool use_bias) {
    Tensor in = create_Tensor(batch, c_in, h, w);
    Tensor filter = create_Tensor(c_out, c_in, 1, 1);
    for (size_t i = 0; i < (size_t) batch * c_in * h * w; ++i) in.data[i] = rand_float();
    for (size_t i = 0; i < (size_t) c_out * c_in; ++i) filter.data[i] = rand_float();

    Float_vector bias;
    vec_init(&bias, c_out);
    for (uint32_t i = 0; i < c_out; ++i) vec_push(&bias, rand_float());
    const Float_vector *bias_ptr = use_bias ? &bias : NULL;

    Tensor ref = conv_1x1_x86(&in, &filter, bias_ptr, stride, relu);
    Tensor neon = conv_1x1_arm_neon(&in, &filter, bias_ptr, stride, relu);

    int ok = ref.batch == neon.batch && ref.channel == neon.channel && ref.height == neon.height &&
             ref.width == neon.width;
    float max_diff = 0.0f;
    if (ok) {
        const size_t n = (size_t) ref.batch * ref.channel * ref.height * ref.width;
        for (size_t i = 0; i < n; ++i) {
            const float d = fabsf(ref.data[i] - neon.data[i]);
            if (d > max_diff) max_diff = d;
        }
        ok = max_diff < 1e-4f;
    }
    if (!ok)
        printf("[FAIL] b=%u c_in=%u c_out=%u %ux%u stride=%u relu=%d bias=%d max_diff=%g\n", batch, c_in, c_out, h,
               w, stride, relu, use_bias, max_diff);

    free_tensor(&in);
    free_tensor(&filter);
    free_tensor(&ref);
    free_tensor(&neon);
    vec_free(&bias);
    return ok;
}

int main(void) {
    srand(12345);
    int failed = 0, total = 0;
    // sizes chosen to hit: pixel tails (plane % 8 != 0), channel tails (c_out % 4 != 0), stride 2, batch > 1
    const uint32_t c_ins[] = {1, 3, 13, 32};
    const uint32_t c_outs[] = {1, 4, 10, 16};
    const uint32_t sizes[][2] = {{1, 1}, {7, 7}, {8, 8}, {9, 5}, {20, 20}};
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            for (int s = 0; s < 5; ++s)
                for (uint32_t stride = 1; stride <= 2; ++stride)
                    for (int relu = 0; relu <= 1; ++relu) {
                        failed += !run_case(s % 2 ? 2 : 1, c_ins[a], c_outs[b], sizes[s][0], sizes[s][1], stride,
                                            relu, (a + b) % 2);
                        ++total;
                    }
    printf("%d/%d cases passed\n", total - failed, total);
    return failed ? 1 : 0;
}
