#define POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h> //using it now but might implement my own in near future
#include <omp.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <stdbool.h>
#ifndef nullptr
#define nullptr NULL
#endif
#ifndef constexpr
#define constexpr const
#endif
// =======> MEMORY ALLOCATION <=======
/*
 * The most important part is not for Windows cause edge devices are mainly arm os
 * Alignment is optimization for ARM NEON registers, ARM NEON on raspberry pi 4b is 16B
 * so 4 float numbers, most optimized will be to work on few registers simultaneously, so alignment
 * to 32B will be good to work on 4 floats at the same time with one cache read
 */
static void *allocate_aligned(size_t alignment, size_t size) {
#if defined(_WIN32) || defined(_WIN64)
    return _aligned_malloc(size, alignment);
#else
    void *ptr = NULL;
    const uint32_t res = posix_memalign(&ptr, alignment, size);
    if (res != 0) {
        fprintf(stderr, "ERROR: posix_memalign returned code %d\n", res);
        return NULL;
    }
    return ptr;
#endif
}

static void free_aligned(void *ptr) {
    if (!ptr) {
        return;
    }
#if defined(_WIN32) || defined(_WIN64)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

//======================================

// =========> FLOAT VECTOR <============
typedef struct {
    float *data;
    size_t capacity;
    size_t size;
} Float_vector;

static void vec_init(Float_vector *v, const size_t init_capacity) {
    v->size = 0;
    v->capacity = init_capacity > 0 ? init_capacity : 4;
    v->data = (float *) malloc(v->capacity * sizeof(float));
    if (v->data == NULL) {
        fprintf(stderr, "ERROR: WRONG VECTOR ALLOCATION\n");
        exit(1);
    }
}

static void vec_push(Float_vector *v, const float val) {
    if (v->size == v->capacity) {
        v->capacity *= 2;
        float *new_data = realloc(v->data, v->capacity * sizeof(float));
        if (new_data == NULL) {
            fprintf(stderr, "ERROR: WRONG VECTOR REALLOCATION\n");
            exit(1);
        }
        v->data = new_data;
    }
    v->data[v->size] = val;
    ++v->size;
}

static void vec_free(Float_vector *v) {
    if (v->data != NULL) {
        free(v->data);
        v->data = nullptr;
    }
    v->size = 0;
    v->capacity = 0;
}

// ===================================

// ============> TENSOR <=============
/*
* tensor shape is [n,c,h,w] (batch size,  number of channels, height, width)
* we want it to be linear for memory so caching would be faster ( no 4 dimensions or more...)
*/
typedef struct {
    uint32_t batch, channel, height, width;
    float *data;
} Tensor;

//calculate index of pixel in location [n,c,h,w] (linear distribution in memory :) )
static size_t get_index_of_pixel(const Tensor *t, const uint32_t n, const uint32_t c, const uint32_t h,
                                 const uint32_t w) {
    return (size_t) n * (t->channel * t->width * t->height) +
           (size_t) c * (t->width * t->height) + (size_t) h * t->width + w;
}

static Tensor create_Tensor(const uint32_t n, const uint32_t c, const uint32_t h, const uint32_t w) {
    Tensor t;
    t.batch = n;
    t.channel = c;
    t.height = h;
    t.width = w;

    //initialize data as zeros
    const size_t numOfPixels = (size_t) n * c * h * w;
    const size_t bytes = numOfPixels * sizeof(float);
    void *ptr = NULL;
    ptr = allocate_aligned(32, bytes);
    if (ptr == NULL) {
        fprintf(stderr, "CRITICAL ERROR: no memory for the tensor :( not good\n");
        exit(1);
    }
    t.data = (float *) (ptr);
    memset(t.data, 0, bytes);
    return t;
}

static void free_tensor(Tensor *t) {
    if (t->data != NULL) {
        free_aligned(t->data);
        t->data = nullptr;
    }
    t->batch = t->channel = t->height = t->width = 0;
}

// ===============================

// ===================> LAYER TYPES <==========================

// ===========> RELU <============
// lets do this in place
// max(0, x)
static Tensor relu(Tensor *t) {
    const size_t total_elements = (size_t) t->batch * t->channel * t->height * t->width;
    for (size_t i = 0; i < total_elements; ++i) {
        if (t->data[i] < 0.0f) {
            t->data[i] = 0.0f;
        }
    }
    const Tensor res = *t;
    t->data = nullptr;
    t->batch = t->channel = t->height = t->width = 0;
    return res;
}

// ===============================

// =========> SIGMOID <===========
// 1 / (1 + e^(-x))
static Tensor sigmoid(Tensor *t) {
    const size_t total_elements = (size_t) t->batch * t->channel * t->height * t->width;
    for (size_t i = 0; i < total_elements; ++i) {
        t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
    }
    const Tensor res = *t;
    t->data = nullptr;
    t->batch = t->channel = t->height = t->width = 0;
    return res;
}

// ===============================

// =========> RESHAPE <===========
static Tensor reshape(Tensor *t, const uint32_t new_n, const uint32_t new_c, const uint32_t new_h,
                      const uint32_t new_w) {
    const size_t total_elements = (size_t) t->batch * t->channel * t->height * t->width;
    const size_t new_total_elements = (size_t) new_n * new_c * new_h * new_w;
    if (new_total_elements != total_elements) {
        fprintf(stderr, "ERROR: Reshape dimensions do not match original size!\n");
        exit(1);
    }
    Tensor res;
    res.batch = new_n;
    res.channel = new_c;
    res.height = new_h;
    res.width = new_w;
    res.data = t->data;

    t->data = nullptr;
    t->batch = t->channel = t->height = t->width = 0;
    return res;
}

// ===============================

// ========> ADD <================
static Tensor add_tensors(const Tensor *t_1, const Tensor *t_2) {
    const Tensor res = create_Tensor(t_1->batch, t_1->channel, t_1->height, t_1->width);
    if (t_1->channel != t_2->channel || t_1->height != t_2->height || t_1->width != t_2->width) {
        fprintf(stderr, "CRITICAL ERROR: Tensors must have the same size to add them!\n");
        exit(1);
    }
    const size_t total_t1 = (size_t) t_1->channel * t_1->batch * t_1->height * t_1->width;
    for (size_t i = 0; i < total_t1; ++i) {
        res.data[i] = t_1->data[i] + t_2->data[i];
    }
    return res;
}

// ===============================

// =======> TRANSPOSE <===========
// change from nhwc to nchw
static Tensor transpose(const Tensor *t) {
    //                              n           c           h               w
    const Tensor res = create_Tensor(t->batch, t->height, t->width, t->channel);
    for (uint32_t b = 0; b < t->batch; ++b) {
        for (uint32_t c = 0; c < t->channel; ++c) {
            for (uint32_t h = 0; h < t->height; ++h) {
                for (uint32_t w = 0; w < t->width; ++w) {
                    const size_t original_indx = get_index_of_pixel(t, b, c, h, w);
                    const size_t after_indx = (size_t) b * (t->height * t->channel * t->width) +
                                              (size_t) h * (t->channel * t->width) +
                                              (size_t) w * t->channel + c;
                    res.data[after_indx] = t->data[original_indx];
                }
            }
        }
    }
    return res;
}

// ===============================

// =======> MAX-POOL_2D <=============
static Tensor maxPool(const Tensor *t, uint32_t pool_size, uint32_t stride) {
    const uint32_t new_h = (t->height - pool_size) / stride + 1;
    const uint32_t new_w = (t->width - pool_size) / stride + 1;
    const Tensor res = create_Tensor(t->batch, t->channel, new_h, new_w);
    for (uint32_t b = 0; b < t->batch; ++b) {
        for (uint32_t c = 0; c < t->channel; ++c) {
            for (uint32_t h = 0; h < new_h; ++h) {
                for (uint32_t w = 0; w < new_w; ++w) {
                    const uint32_t res_indx = get_index_of_pixel(&res, b, c, h, w);
                    float max_val = -INFINITY;
                    for (uint32_t i = 0; i < pool_size; ++i) {
                        for (uint32_t j = 0; j < pool_size; ++j) {
                            const uint32_t t_i = (h * stride) + i; //columna ten
                            const uint32_t t_j = (w * stride) + j; //row
                            const uint32_t t_index = get_index_of_pixel(t, b, c, t_i, t_j);
                            if (t->data[t_index] > max_val) {
                                max_val = t->data[t_index];
                            }
                        }
                    }
                    res.data[res_indx] = max_val;
                }
            }
        }
    }
    return res;
}

// ===============================

// ==========> RESIZE <===========
// yunet resize is asymmetric and algorithm is nearest neigbour with floor variant so first we make new tensor,
// and for every pixel we do simple operation old_tensor_id = floor(new_tensor_index/ scale)
// resize for channel and batches are left 1:1 hopefully...
// ===============================
static Tensor resize_nn_asymmetric(const Tensor *t, const uint32_t new_width, const uint32_t new_height) {
    const Tensor res = create_Tensor(t->batch, t->channel, new_height, new_width);
    const float scale_x = (float) new_width / (float) t->width;
    const float scale_y = (float) new_height / (float) t->height;
    for (uint32_t b = 0; b < t->batch; ++b) {
        for (uint32_t c = 0; c < t->channel; ++c) {
            for (uint32_t y = 0; y < new_height; ++y) {
                for (uint32_t x = 0; x < new_width; ++x) {
                    uint32_t old_y = (uint32_t) (y / scale_y);
                    uint32_t old_x = (uint32_t) (x / scale_x);

                    if (old_y >= t->height) old_y = t->height - 1;
                    if (old_x >= t->width) old_x = t->width - 1;

                    const size_t new_tensor_index = get_index_of_pixel(&res, b, c, y, x);
                    const size_t old_tensor_index = get_index_of_pixel(t, b, c, old_y, old_x);

                    res.data[new_tensor_index] = t->data[old_tensor_index];
                }
            }
        }
    }
    return res;
}

// ===============================

// ====> Batch Normalization <====
// ===============================

// ===========> PRelu <===========
static Tensor PRelu(Tensor *t, const Float_vector* slope ) {
    const size_t planeSize = t->height * t->width;
    for (uint32_t b = 0; b < t->batch; ++b) {
        for (uint32_t c = 0; c < t->channel; ++c) {
            float s = slope->data[c];
            float *ptr = t->data + (b * t->channel + c) * planeSize;
            for (size_t i = 0; i < planeSize; ++i) {
                if (ptr[i]<0.0f) ptr[i] *= s;
            }
        }
    }
    const Tensor res = *t;
    t->data = nullptr;
    t->batch = t->channel = t->height = t->width = 0;
    return res;
}
// ===============================
// ========> GEMM <==============
// GEMM: out[i] = bias[i] + sum_j(weight[i*in + j] * input[j])
// weight shape: [out_features, in_features], np. [128, 50176]
static Tensor GEMM(const Tensor *t, const Tensor *weight, const Float_vector *bias ) {
    const uint32_t out_features = weight->batch;
    const uint32_t in_features = weight->channel;
    Tensor res = create_Tensor(1, out_features, 1, 1);
#pragma omp parallel for
    for (uint32_t i = 0; i<out_features; ++i) {
        const float* w_row = weight->data + i*in_features;
        float sum = bias && bias->data ? bias->data[i] : 0.0f;
        for (uint32_t j = 0; j<in_features; ++j) {
            sum += w_row[j]*t->data[j];
        }
        res.data[i] = sum;
    }
    return res;
}

// ===============================
// =======> Flatten <=============
// N,C,H,W -> N, C*H*W, 1, 1
static Tensor flatten(Tensor *t) {
    const uint32_t flat_size = t->channel*t->height*t->width;
    Tensor res;
    res.batch = t->batch;
    res.channel = flat_size;
    res.height = 1;
    res.width = 1;
    res.data=t->data;
    t->data = nullptr;
    t->batch = t->channel = t->height = t->width = 0;
    return res;
}
// ===============================

// =========> subtract Scalar <==========
static Tensor sub_scalar(Tensor *t, const float sub) {
    const size_t total_elements = (size_t) t->batch * t->channel * t->height * t->width;
    for (size_t i = 0; i < total_elements; ++i) {
        t->data[i]-=sub;
    }
    const Tensor res = *t;
    t->data = nullptr;
    t->batch = t->channel = t->height = t->width = 0;
    return res;
}
// ===============================

// ========> multiply scalar <===========
static Tensor mul_scalar(Tensor *t,const float mul) {
    const size_t total_elements = (size_t) t->batch * t->channel * t->height * t->width;
    for (size_t i = 0; i < total_elements; ++i) {
        t->data[i]*=mul;
    }
    const Tensor res = *t;
    t->data = nullptr;
    t->batch = t->channel = t->height = t->width = 0;
    return res;
}
// ===============================

// ==========> CONV <=============
//classical conv
// out size formula is same for height and width O= floor((I-K+2P)/S) + 1, where I is input size, K is kernel size,
// p is padding and S is Stride
//  filter tensor params are [C_out, C_in, K_h, K_w] - c_out is number of channels on out tensor and C_in is channels _in
static Tensor conv(const Tensor *input, const Tensor *filter_tensor, const Float_vector *bias, const uint32_t stride,
                   const uint32_t padding, bool apply_relu) {
    //more clean names
    const uint32_t out_channels = filter_tensor->batch; // number of filters
    const uint32_t in_channels = filter_tensor->channel; // filter Z coordinate
    const uint32_t kernel_h = filter_tensor->height; // filter Y coordinate
    const uint32_t kernel_w = filter_tensor->width; // filter X coordinate

    if (input->channel != in_channels) {
        fprintf(stderr, "CRITICAL ERROR: Input channels (%u) do not match weight in_channels (%u)!\n", input->channel,
                in_channels);
        exit(1);
    }

    const uint32_t out_h = (input->height - kernel_h + 2 * padding) / stride + 1;
    const uint32_t out_w = (input->width - kernel_w + 2 * padding) / stride + 1;
    const Tensor out = create_Tensor(input->batch, out_channels, out_h, out_w);

    const size_t out_channel_size = (size_t) out_h * out_w;
    const size_t input_channel_size = input->height * input->width;

    for (uint32_t b = 0; b < input->batch; ++b) {
#pragma omp parallel for
        for (uint32_t c = 0; c < out_channels; ++c) {
            float *out_c_pointer = out.data + b * out_channels * out_channel_size + c * out_channel_size;
            const float bias_val = bias != NULL && bias->size > c ? bias->data[c] : 0.0f;
            const float *filter_pointer = filter_tensor->data + c * in_channels * kernel_h * kernel_w;

            for (uint32_t y = 0; y < out_h; ++y) {
                for (uint32_t x = 0; x < out_w; ++x) {
                    float sum = bias_val;
                    const float *weight_pointer = filter_pointer;
                    // we go through every filter tensor and tensor has parameters [ic,kh,kw] b in filter tensor is numvber of tensors
                    for (uint32_t ic = 0; ic < in_channels; ++ic) {
                        const float *input_channel_ptr =
                                input->data + b * in_channels * input_channel_size + (ic * input_channel_size);
                        for (uint32_t ky = 0; ky < kernel_h; ++ky) {
                            //if padding we skip (do not bother multiplying with 0)
                            const int32_t in_y = (y * stride) + ky - padding;
                            if (in_y < 0 || in_y >= (int32_t) input->height) {
                                weight_pointer += kernel_w;
                                continue;
                            }
                            for (uint32_t kx = 0; kx < kernel_w; ++kx) {
                                const int32_t in_x = (x * stride) + kx - padding;
                                //check if it is not padding (zeros)
                                if (in_x >= 0 && in_x < (int32_t) input->width) {
                                    const size_t in_idx = in_y * input->width + in_x;
                                    sum += input_channel_ptr[in_idx] * *weight_pointer;
                                }
                                weight_pointer++;
                            }
                        }
                    }
                    if (apply_relu && sum < 0.0f) {
                        sum = 0.0f;
                    }
                    *out_c_pointer = sum;
                    out_c_pointer++;
                }
            }
        }
    }
    return out;
}

// ===============================

// =======> DepthWise Conv <======
static Tensor depthwise_conv2D(const Tensor *input, const Tensor *filter_tensor, const Float_vector *bias,
                               const uint32_t stride,
                               const uint32_t padding, bool apply_relu) {
    // In Depthwise conv weights parameters are [number of channels, 1, kernel_h,kernel_w]
    const uint32_t channels = filter_tensor->batch; // number of filters
    const uint32_t kernel_h = filter_tensor->height; // filter Y coordinate
    const uint32_t kernel_w = filter_tensor->width; // filter X coordinate

    if (input->channel != channels) {
        fprintf(stderr, "CRITICAL ERROR: Input channels (%u) do not match weight filters (%u)!\n", input->channel,
                channels);
        exit(1);
    }

    const uint32_t out_h = (input->height - kernel_h + 2 * padding) / stride + 1;
    const uint32_t out_w = (input->width - kernel_w + 2 * padding) / stride + 1;
    const Tensor out = create_Tensor(input->batch, channels, out_h, out_w);

    const size_t input_channel_size = input->height * input->width;
    const size_t filter_size = kernel_h * kernel_w;
    const size_t out_channel_size = (size_t) out_h * out_w;

    for (uint32_t b = 0; b < input->batch; ++b) {
#pragma omp parallel for
        for (uint32_t c = 0; c < channels; ++c) {
            float *out_c_pointer = out.data + b * channels * out_channel_size + c * out_channel_size;
            const float bias_val = bias != NULL && bias->size > c ? bias->data[c] : 0.0f;
            const float *filter_base_ptr = filter_tensor->data + c * filter_size;
            const float *input_channel_ptr = input->data + b * channels * input_channel_size + c * input_channel_size;

            for (uint32_t y = 0; y < out_h; ++y) {
                for (uint32_t x = 0; x < out_w; ++x) {
                    float sum = bias_val;
                    const float *w_ptr = filter_base_ptr;
                    // we go through every filter tensor (2 dimensions not 3 like standard conv [kw,kh]
                    // b in filter tensor is number of tensor
                    for (uint32_t ky = 0; ky < kernel_h; ++ky) {
                        const int32_t in_y = y * stride + ky - padding;
                        if (in_y < 0 || in_y >= (int32_t) input->height) {
                            w_ptr += kernel_w;
                            continue;
                        }
                        for (uint32_t kx = 0; kx < kernel_w; ++kx) {
                            //calculate parameters of input tensor
                            const int32_t in_x = x * stride + kx - padding;
                            //check if it is not padding (zeros)
                            if (in_x >= 0 && in_x < (int32_t) input->width) {
                                const size_t input_indx_x = in_y * input->width + in_x;
                                sum += input_channel_ptr[input_indx_x] * *w_ptr;
                            }
                            w_ptr++;
                        }
                    }
                    if (apply_relu && sum < 0.0f) {
                        sum = 0.0f;
                    }

                    *out_c_pointer = sum;
                    out_c_pointer++;
                }
            }
        }
    }
    return out;
}

// ===============================


// ===========>  FORWARD PASS <==============
typedef enum {
    LAYER_CONV,
    LAYER_DEPTHWISE_CONV,
    LAYER_MAXPOOL,
    LAYER_RESIZE,
    LAYER_TRANSPOSE,
    LAYER_ADD,
    LAYER_RESHAPE,
    LAYER_SIGMOID,
    LAYER_RELU,
    LAYER_PRELU,
    LAYER_FLATTEN,
    LAYER_GEMM,
    LAYER_SUB_SCALAR,
    LAYER_MUL_SCALAR,
} LayerType;

//what .cfg file can have (for more check build_nn_graph.py)
typedef struct {
    LayerType type;
    int32_t id;

    //indexes of input neurons
    //in yunet only neuron ADD has 2 input neurons :3
    int32_t input1_idx;
    int32_t input2_idx;

    // kernels sizes
    uint32_t kernel_h;
    uint32_t kernel_w;

    uint32_t stride;
    uint32_t padding;
    //optimization for conv - conv+relu simultaneously
    bool apply_relu;
    float scalar_val;

    // Dimensions for Reshape / Resize
    uint32_t out_n, out_c, out_h, out_w;


    // weights for conv layers
    Tensor weights;
    Float_vector bias;

    // how many times neuron is used as input
    int32_t ref_count;
    char out_name[32]; // e.g. "cls_8", "bbox_16", or ""
} Layer;

typedef struct {
    Layer *layers;
    size_t num_layers;
} Network;

typedef struct {
    char name[32]; // "cls_8", "bbox_16",   "fc1"
    Tensor tensor;
} OutputTensor;

// for example: yunet has 12 outputs and sface has 1 output
typedef struct {
    OutputTensor *outputs;
    size_t count;
} NetworkOutputs;

static void free_network_outputs(NetworkOutputs *outs) {
    for (size_t i = 0; i < outs->count; ++i) {
        free_tensor(&outs->outputs[i].tensor);
    }
    if (outs->outputs != NULL) {
        free(outs->outputs);
        outs->outputs = nullptr;
    }
    outs->count = 0;
}

static void free_network(Network *net) {
    for (size_t i = 0; i < net->num_layers; ++i) {
        if (net->layers[i].type == LAYER_CONV ||
            net->layers[i].type == LAYER_DEPTHWISE_CONV ||
            net->layers[i].type == LAYER_GEMM) {
            free_tensor(&net->layers[i].weights);
            vec_free(&net->layers[i].bias);
            } else if (net->layers[i].type == LAYER_PRELU) {
                vec_free(&net->layers[i].bias);
            }
    }
    if (net->layers) {
        free(net->layers);
        net->layers = nullptr;
    }
    net->num_layers = 0;
}

static Tensor load_weights(const char *filepath, const uint32_t n, const uint32_t c, const uint32_t h, const uint32_t w) {
    const Tensor t = create_Tensor(n, c, h, w);
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "CRITICAL ERROR: Cannot open weight file %s\n", filepath);
        exit(1);
    }
    const size_t total_floats = (size_t) n * c * h * w;
    const size_t read_count = fread(t.data, sizeof(float), total_floats, f);
    if (read_count != total_floats) {
        fprintf(stderr, "ERROR: Expected %zu floats in %s, read %zu\n", total_floats, filepath, read_count);
    }
    fclose(f);
    return t;
}

static void load_bias(const char *filepath, Float_vector *vec, uint32_t size) {
    vec_init(vec, size);
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "CRITICAL ERROR: Cannot open bias file %s\n", filepath);
        exit(1);
    }
    const size_t read_count = fread(vec->data, sizeof(float), size, f);
    if (read_count != size) {
        fprintf(stderr, "ERROR: Expected %u bias floats in %s, read %zu\n", size, filepath, read_count);
    }
    vec->size = size;
    fclose(f);
}

static Network load_network(const char *cfg_path, const char *weights_dir) {
    // loading data
    FILE *f = fopen(cfg_path, "r");
    char final_weights_dir[256];
    strncpy(final_weights_dir, weights_dir, sizeof(final_weights_dir));

    if (!f) {
        char fallback_cfg[512];
        snprintf(fallback_cfg, sizeof(fallback_cfg), "../%s", cfg_path);
        f = fopen(fallback_cfg, "r");
        if (f) {
            snprintf(final_weights_dir, sizeof(final_weights_dir), "../%s", weights_dir);
        }
    }

    if (!f) {
        fprintf(stderr, "CRITICAL ERROR: Cannot open cfg file %s (also tried ../%s)\n", cfg_path, cfg_path);
        exit(1);
    }

    size_t num_layers = 0;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (strlen(line) > 3) num_layers++;
    }
    //reset pointer
    rewind(f);

    Network net;
    net.num_layers = num_layers;
    net.layers = (Layer *) malloc(num_layers * sizeof(Layer));
    memset(net.layers, 0, num_layers * sizeof(Layer));

    size_t idx = 0;
    while (fgets(line, sizeof(line), f) && idx < num_layers) {
        if (strlen(line) <= 3) continue;

        Layer *l = &net.layers[idx];
        l->input1_idx = -1;
        l->input2_idx = -1;

        char type_str[32];
        sscanf(line, "%d %31s", &l->id, type_str);

        if (strcmp(type_str, "CONV") == 0) l->type = LAYER_CONV;
        else if (strcmp(type_str, "DEPTHWISE") == 0) l->type = LAYER_DEPTHWISE_CONV;
        else if (strcmp(type_str, "MAXPOOL") == 0) l->type = LAYER_MAXPOOL;
        else if (strcmp(type_str, "RESIZE") == 0) l->type = LAYER_RESIZE;
        else if (strcmp(type_str, "ADD") == 0) l->type = LAYER_ADD;
        else if (strcmp(type_str, "TRANSPOSE") == 0) l->type = LAYER_TRANSPOSE;
        else if (strcmp(type_str, "RESHAPE") == 0) l->type = LAYER_RESHAPE;
        else if (strcmp(type_str, "SIGMOID") == 0) l->type = LAYER_SIGMOID;
        else if (strcmp(type_str, "RELU") == 0) l->type = LAYER_RELU;
        else if (strcmp(type_str, "PRELU") == 0) l->type = LAYER_PRELU;
        else if (strcmp(type_str, "FLATTEN") == 0) l->type = LAYER_FLATTEN;
        else if (strcmp(type_str, "GEMM") == 0) l->type = LAYER_GEMM;
        else if (strcmp(type_str, "SUB_SCALAR") == 0) l->type = LAYER_SUB_SCALAR;
        else if (strcmp(type_str, "MUL_SCALAR") == 0) l->type = LAYER_MUL_SCALAR;

        char *p;
        if ((p = strstr(line, "inputs="))) {
            char in_str[64];
            sscanf(p + 7, "%63s", in_str);
            if (strchr(in_str, ',')) {
                sscanf(in_str, "%d,%d", &l->input1_idx, &l->input2_idx);
            } else {
                sscanf(in_str, "%d", &l->input1_idx);
            }
        }
        if ((p = strstr(line, "scalar="))) sscanf(p + 7, "%f", &l->scalar_val);
        if ((p = strstr(line, "stride="))) sscanf(p + 7, "%u", &l->stride);
        if ((p = strstr(line, "pad="))) sscanf(p + 4, "%u", &l->padding);
        int relu_int = 0;
        if ((p = strstr(line, "relu="))) {
            sscanf(p + 5, "%d", &relu_int);
            l->apply_relu = (relu_int == 1);
        }
        if ((p = strstr(line, "kernel="))) {
            sscanf(p + 7, "%u", &l->kernel_h);
            l->kernel_w = l->kernel_h;
        }
        if ((p = strstr(line, "ref_count="))) sscanf(p + 10, "%d", &l->ref_count);
        if ((p = strstr(line, "out_name="))) sscanf(p + 9, "%31s", l->out_name);

        if ((p = strstr(line, "out_shape="))) {
            char s_buf[64];
            sscanf(p + 10, "%63s", s_buf);
            uint32_t d1 = 1, d2 = 1, d3 = 1, d4 = 1;
            const int count = sscanf(s_buf, "%u,%u,%u,%u", &d1, &d2, &d3, &d4);
            if (count == 4) {
                l->out_n = d1;
                l->out_c = d2;
                l->out_h = d3;
                l->out_w = d4;
            } else if (count == 3) {
                l->out_n = d1;
                l->out_c = 1;
                l->out_h = d2;
                l->out_w = d3;
            }
        }

        if (l->type == LAYER_CONV || l->type == LAYER_DEPTHWISE_CONV || l->type == LAYER_GEMM) {
            char w_file[128] = {0};
            char b_file[128] = {0};
            if ((p = strstr(line, "weights="))) sscanf(p + 8, "%127s", w_file);
            if ((p = strstr(line, "bias="))) sscanf(p + 5, "%127s", b_file);

            const char *ws = strstr(line, "w_shape=");
            if (ws && strcmp(w_file, "NULL") != 0) {
                uint32_t wn, wc, wh, ww;
                sscanf(ws + 8, "%u,%u,%u,%u", &wn, &wc, &wh, &ww);
                char path[256];
                snprintf(path, sizeof(path), "%s/%s", final_weights_dir, w_file);
                l->weights = load_weights(path, wn, wc, wh, ww);
            }

            if (strcmp(b_file, "NULL") != 0) {
                char path[256];
                snprintf(path, sizeof(path), "%s/%s", final_weights_dir, b_file);
                load_bias(path, &l->bias, l->weights.batch);
            }
        }
        if (l->type == LAYER_PRELU) {
            char b_file[128] = {0};
            uint32_t slope_len = 0;
            if ((p = strstr(line, "bias="))) sscanf(p + 5, "%127s", b_file);
            if ((p = strstr(line, "slope_len="))) sscanf(p + 10, "%u", &slope_len);
            if (slope_len > 0 && strcmp(b_file, "NULL") != 0) {
                char path[256];
                snprintf(path, sizeof(path), "%s/%s", final_weights_dir, b_file);
                load_bias(path, &l->bias, slope_len);
            }
        }
        idx++;
    }
    fclose(f);
    return net;
}

static NetworkOutputs forward_network(const Network *net, const Tensor *input) {
    //potential optimization but for yunet it is only 2KB so it is ok i think
    Tensor *outputs = malloc(net->num_layers * sizeof(Tensor));
    if (!outputs) {
        fprintf(stderr, "CRITICAL ERROR: Memory allocation failed for outputs array\n");
        exit(1);
    }
    // array for dynamic deletion of tensor (it is keeping number of references for each tensor)
    int32_t *live_refs = (int32_t *) malloc(net->num_layers * sizeof(int32_t));
    for (size_t k = 0; k < net->num_layers; ++k) {
        live_refs[k] = net->layers[k].ref_count;
    }

    for (size_t i = 0; i < net->num_layers; ++i) {
        const Layer *layer = &net->layers[i];
        //check if it is input layer
        Tensor *in1 = layer->input1_idx == -1 ? (Tensor *) input : &outputs[layer->input1_idx];
        const Tensor *in2 = layer->input2_idx != -1 ? &outputs[layer->input2_idx] : nullptr;

        switch (layer->type) {
            case LAYER_CONV:
                outputs[i] = conv(in1, &layer->weights, &layer->bias, layer->stride, layer->padding, layer->apply_relu);
                break;
            case LAYER_DEPTHWISE_CONV:
                outputs[i] = depthwise_conv2D(in1, &layer->weights, &layer->bias, layer->stride, layer->padding,
                                              layer->apply_relu);
                break;
            case LAYER_MAXPOOL:
                outputs[i] = maxPool(in1, layer->kernel_h, layer->stride);
                break;
            case LAYER_RESIZE:
                outputs[i] = resize_nn_asymmetric(in1, layer->out_w, layer->out_h);
                break;
            case LAYER_ADD:
                outputs[i] = add_tensors(in1, in2);
                break;
            case LAYER_TRANSPOSE:
                outputs[i] = transpose(in1);
                break;
            case LAYER_RESHAPE:
                outputs[i] = reshape(in1, layer->out_n, layer->out_c, layer->out_h, layer->out_w);
                break;
            case LAYER_SIGMOID:
                outputs[i] = sigmoid(in1);
                break;
            case LAYER_RELU:
                outputs[i] = relu(in1);
                break;
            case LAYER_PRELU:
                outputs[i] = PRelu(in1, &layer->bias);
                break;
            case LAYER_FLATTEN:
                outputs[i] = flatten(in1);
                break;
            case LAYER_GEMM:
                outputs[i] = GEMM(in1, (Tensor*)&layer->weights, (Float_vector*)&layer->bias);
                break;
            case LAYER_SUB_SCALAR:
                outputs[i] = sub_scalar(in1, layer->scalar_val);
                break;
            case LAYER_MUL_SCALAR:
                outputs[i] = mul_scalar(in1, layer->scalar_val);
                break;
            default:
                fprintf(stderr, "CRITICAL ERROR: Unsupported layer type %d at index %zu\n", layer->type, i);
                exit(1);
        }
        if (layer->input1_idx != -1) {
            live_refs[layer->input1_idx]--;
            if (live_refs[layer->input1_idx] == 0) {
                free_tensor(&outputs[layer->input1_idx]);
            }
        }
    }
    size_t out_count = 0;
    for (size_t i = 0; i < net->num_layers; ++i) {
        if (net->layers[i].ref_count == 0) {
            out_count++;
        }
    }

    NetworkOutputs net_out;
    net_out.count = out_count;
    net_out.outputs = (OutputTensor *) malloc(out_count * sizeof(OutputTensor));

    size_t out_idx = 0;
    for (size_t i = 0; i < net->num_layers; ++i) {
        if (net->layers[i].ref_count == 0) {
            strncpy(net_out.outputs[out_idx].name, net->layers[i].out_name, 31);
            net_out.outputs[out_idx].name[31] = '\0';
            net_out.outputs[out_idx].tensor = outputs[i];
            out_idx++;
        } else {
            free_tensor(&outputs[i]);
        }
    }

    free(outputs);
    free(live_refs);
    return net_out;
}

// ==================> Face objects <============
typedef struct {
    float x, y, w, h; // [x,y,width,height] of bbox (x y are center of bbox not lef top corner)
    float landmarks[10]; // [x1,y2  x2,y2...] 5 face point left\right eye, nose , left/right side of mouth
    float confidence; //result of cls*obj (confidence there is a face * quality of bounding box)
    bool isRemoved; //true after nms
} Face;

typedef struct {
    Face *faces;
    size_t size;
    size_t capacity;
} FaceVector;

static void face_vec_init(FaceVector *v, const size_t init_capacity) {
    v->size = 0;
    v->capacity = (init_capacity > 0) ? init_capacity : 16;
    v->faces = (Face *) malloc(v->capacity * sizeof(Face));
    if (v->faces == NULL) {
        fprintf(stderr, "ERROR: WRONG FACE VECTOR ALLOCATION\n");
        exit(1);
    }
}

static void face_vec_push(FaceVector *v, Face f) {
    if (v->size == v->capacity) {
        v->capacity *= 2;
        Face *new_faces = (Face *) realloc(v->faces, v->capacity * sizeof(Face));
        if (new_faces == NULL) {
            fprintf(stderr, "ERROR: WRONG FACE VECTOR ALLOCATION\n");
            exit(1);
        }
        v->faces = new_faces;
    }
    v->faces[v->size] = f;
    ++v->size;
}

static void face_vec_free(FaceVector *v) {
    if (v->faces != NULL) {
        free(v->faces);
        v->faces = nullptr;
    }
    v->capacity = 0;
    v->size = 0;
}

static const Tensor *find_output_tensor(const NetworkOutputs *outs, const char *name) {
    for (size_t k = 0; k < outs->count; ++k) {
        if (strcmp(outs->outputs[k].name, name) == 0) {
            return &outs->outputs[k].tensor;
        }
    }
    return nullptr;
}

// ========> Yunet Decoding <============
static void decode_yunet(const NetworkOutputs *outs, FaceVector *faces,
                         const uint32_t input_height, const uint32_t input_width, const float score_treshold) {
    uint32_t strides[3] = {8, 16, 32};
    const float score_tresh_sq = score_treshold * score_treshold;

    for (int s = 0; s < 3; ++s) {
        const uint32_t stride = strides[s];
        char cls_name[16], obj_name[16], bbox_name[16], kps_name[16];
        snprintf(cls_name, sizeof(cls_name), "cls_%u", stride);
        snprintf(obj_name, sizeof(obj_name), "obj_%u", stride);
        snprintf(bbox_name, sizeof(bbox_name), "bbox_%u", stride);
        snprintf(kps_name, sizeof(kps_name), "kps_%u", stride);

        //GET OUTPUT FOR EVERY STRIDE
        const Tensor *cls_t = find_output_tensor(outs, cls_name);
        const Tensor *obj_t = find_output_tensor(outs, obj_name);
        const Tensor *bbox_t = find_output_tensor(outs, bbox_name);
        const Tensor *kps_t = find_output_tensor(outs, kps_name);

        if (!cls_t || !obj_t || !bbox_t || !kps_t) {
            fprintf(stderr, "WARNING: Missing heads for stride %u\n", stride);
            continue;
        }
        // how many potential faces for every stride
        const uint32_t columns = input_width / stride;
        const uint32_t rows = input_height / stride;
        const float *cls_ptr = cls_t->data;
        const float *obj_ptr = obj_t->data;
        const float *bbox_ptr = bbox_t->data;
        const float *kps_ptr = kps_t->data;

        for (uint32_t r = 0; r < rows; ++r) {
            for (uint32_t c = 0; c < columns; ++c) {
                size_t indx = r * columns + c;

                float cls = cls_ptr[indx];
                if (cls < score_tresh_sq) continue;
                if (cls > 1.0f)cls = 1.0f;

                float obj = obj_ptr[indx];
                if (obj <= 0.0f) continue;
                if (obj > 1.0f) obj = 1.0f;


                const float scr = cls * obj;
                if (scr < score_tresh_sq) continue;

                const float score = sqrtf(scr);

                //decode bbox (every bbox has 4 parameters)
                const float *b = &bbox_ptr[indx * 4];
                const float center_x = ((float) c + b[0]) * (float) stride;
                const float center_y = ((float) r + b[1]) * (float) stride;
                //remember that r and c are indexes of these small squares
                const float w = expf(b[2]) * (float) stride;
                const float h = expf(b[3]) * (float) stride;

                Face face;
                face.confidence = score;
                face.x = center_x - w * 0.5f;
                face.y = center_y - h * 0.5f;
                face.w = w;
                face.h = h;
                face.isRemoved = false;

                //decode landmarks
                const float *kps = &kps_ptr[indx * 10];
                for (uint32_t k = 0; k < 5; ++k) {
                    face.landmarks[k * 2] = (kps[k * 2] + (float) c) * (float) stride;
                    face.landmarks[k * 2 + 1] = (kps[k * 2 + 1] + (float) r) * (float) stride;
                }
                face_vec_push(faces, face);
            }
        }
    }
}

// ============> NMS - leave only real faces with highest possibility <========
//calculate how much lider's bounding box is in this b bounding box if more than threshold it is removed, if not it is other face
static float calculate_iou(const Face *a, const Face *b) {
    // get union area of 2 possible faces
    const float x1 = a->x > b->x ? a->x : b->x;
    const float y1 = a->y > b->y ? a->y : b->y;
    const float x2 = a->x + a->w < b->x + b->w ? a->x + a->w : b->x + b->w;
    const float y2 = a->y + a->h < b->y + b->h ? a->y + a->h : b->y + b->h;
    const float inter_w = x2 - x1;
    const float inter_h = y2 - y1;
    if (inter_w <= 0.0f || inter_h <= 0.0f) return 0.0f;

    const float inter_area = inter_w * inter_h;
    const float union_area = a->w * a->h + b->w * b->h - inter_area;

    return union_area > 0.0f ? inter_area / union_area : 0.0f;
}

static int compare_faces_desc(const void *a, const void *b) {
    const Face *fa = (const Face *) a;
    const Face *fb = (const Face *) b;
    if (fb->confidence > fa->confidence) return 1;
    if (fb->confidence < fa->confidence) return -1;
    return 0;
}

static void nms(FaceVector *faces_vector, const float nms_thresh) {
    if (faces_vector->size <= 1) return;
    qsort(faces_vector->faces, faces_vector->size, sizeof(Face), compare_faces_desc);

    for (size_t i = 0; i < faces_vector->size; ++i) {
        if (faces_vector->faces[i].isRemoved) continue;
        for (size_t j = i + 1; j < faces_vector->size; ++j) {
            if (faces_vector->faces[j].isRemoved) continue;
            if (calculate_iou(&faces_vector->faces[i], &faces_vector->faces[j]) > nms_thresh) {
                faces_vector->faces[j].isRemoved = true;
            }
        }
    }
    //compact vector
    size_t write_idx = 0;
    for (size_t i = 0; i < faces_vector->size; ++i) {
        if (!faces_vector->faces[i].isRemoved) {
            if (write_idx != i) {
                faces_vector->faces[write_idx] = faces_vector->faces[i];
            }
            write_idx++;
        }
    }
    faces_vector->size = write_idx;
}

// =============> Face Alignment - Kabsh-Umeyama Algorithm <===============
static const float REFERENCE_FACIAL_POINTS[5][2] = {
    {38.2946f, 51.6963f}, // left eye
    {73.5318f, 51.6963f}, // right eye
    {55.9020f, 71.7366f}, // nose
    {41.5493f, 92.3655f}, // left mouth
    {70.7299f, 92.3655f} // right mouth
};

typedef struct {
    float m[2][3];
} AffineMatrix_2x3;

static AffineMatrix_2x3 calculate_affine_matrix(const float src_landmarks[10]) {
    // calculate centroids
    float src_mean_x = 0.0f, src_mean_y = 0.0f;
    float dst_mean_x = 0.0f, dst_mean_y = 0.0f;
    for (int i = 0; i < 5; ++i) {
        src_mean_x += src_landmarks[i * 2 + 0];
        src_mean_y += src_landmarks[i * 2 + 1];
        dst_mean_x += REFERENCE_FACIAL_POINTS[i][0];
        dst_mean_y += REFERENCE_FACIAL_POINTS[i][1];
    }
    src_mean_x /= 5.0f;
    src_mean_y /= 5.0f;
    dst_mean_x /= 5.0f;
    dst_mean_y /= 5.0f;

    // Variance and cross-Variance
    float var_x = 0.0f; // source variance
    float A_dot = 0.0f; // scalar of source and dest
    float B_dot = 0.0f; // vector of source and dest

    for (int i = 0; i < 5; ++i) {
        const float sx = src_landmarks[i * 2 + 0] - src_mean_x;
        const float sy = src_landmarks[i * 2 + 1] - src_mean_y;
        const float dx = REFERENCE_FACIAL_POINTS[i][0] - dst_mean_x;
        const float dy = REFERENCE_FACIAL_POINTS[i][1] - dst_mean_y;

        var_x += sx * sx + sy * sy;
        A_dot += dx * sx + dy * sy;
        B_dot += dy * sx - dx * sy;
    }
    if (var_x < 1e-6f) var_x = 1e-6f;

    const float a = A_dot / var_x;
    const float b = B_dot / var_x;
    const float tx = dst_mean_x - (a * src_mean_x - b * src_mean_y);
    const float ty = dst_mean_y - (b * src_mean_x + a * src_mean_y);

    AffineMatrix_2x3 M;
    M.m[0][0] = a;
    M.m[0][1] = -b;
    M.m[0][2] = tx;
    M.m[1][0] = b;
    M.m[1][1] = a;
    M.m[1][2] = ty;
    return M;
}

// Warp Affine
static Tensor align_crop_face(const unsigned char *src_img, const int img_w, const int img_h, float src_landmarks[10]) {
    //create input tensor for sFace
    constexpr uint32_t out_w = 112;
    constexpr uint32_t out_h = 112;
    const Tensor aligned = create_Tensor(1, 3, out_h, out_w);
    constexpr size_t plane_size = (size_t) out_h * out_w;
    // M = |a -b tx|
    //     |b  a ty|
    const AffineMatrix_2x3 M = calculate_affine_matrix(src_landmarks);
    const float a = M.m[0][0];
    const float b = M.m[1][0];
    const float tx = M.m[0][2];
    const float ty = M.m[1][2];
    //calculate determinant
    float det = a * a + b * b;
    if (det < 1e-8f) det = 1e-8f;
    // u = a*x - b*y + tx   v = b*x - a*x + ty
    // we know u and v so we want to get x and y
    // x = (a * (u -tx) + b * (v-ty))/det   y = (-b * (u -tx) + a * (v -ty)) / det
    for (uint32_t v = 0; v < out_h; ++v) {
        for (uint32_t u = 0; u < out_w; ++u) {
            //inverse mapping [u,v] = [src_x,src_y]
            const float src_x = (a * (u - tx) + b * (v - ty)) / det;
            const float src_y = (-b * (u - tx) + a * (v - ty)) / det;

            const int x0 = (int) floorf(src_x);
            const int y0 = (int) floorf(src_y);
            const int x1 = x0 + 1;
            const int y1 = y0 + 1;

            const float fx = src_x - (float) x0;
            const float fy = src_y - (float) y0;

            float r = 0.0f, g = 0.0f, bl = 0.0f;

            if (x0 >= 0 && x1 < img_w && y0 >= 0 && y1 < img_h) {
                const float w00 = (1.0f - fx) * (1.0f - fy);
                const float w10 = fx * (1.0f - fy);
                const float w01 = (1.0f - fx) * fy;
                const float w11 = fx * fy;

                const int idx00 = (y0 * img_w + x0) * 3;
                const int idx10 = (y0 * img_w + x1) * 3;
                const int idx01 = (y1 * img_w + x0) * 3;
                const int idx11 = (y1 * img_w + x1) * 3;

                r = w00 * src_img[idx00 + 0] + w10 * src_img[idx10 + 0] + w01 * src_img[idx01 + 0] + w11 * src_img[
                        idx11 + 0];
                g = w00 * src_img[idx00 + 1] + w10 * src_img[idx10 + 1] + w01 * src_img[idx01 + 1] + w11 * src_img[
                        idx11 + 1];
                bl = w00 * src_img[idx00 + 2] + w10 * src_img[idx10 + 2] + w01 * src_img[idx01 + 2] + w11 * src_img[
                         idx11 + 2];
            }

            const size_t out_idx = v * out_w + u;

            aligned.data[0 * plane_size + out_idx] = bl;
            aligned.data[1 * plane_size + out_idx] = g;
            aligned.data[2 * plane_size + out_idx] = r;
        }
    }
    return aligned;
}

static Tensor load_image_as_tensor(const char *filepath, const uint32_t target_w, const uint32_t target_h) {
    int img_w, img_h, channels;
    // load RGB
    unsigned char *raw_data = stbi_load(filepath, &img_w, &img_h, &channels, 3);
    if (!raw_data) {
        fprintf(stderr, "ERROR: Cannot load image file '%s'\n", filepath);
        exit(1);
    }
    printf("[+] Image '%s' loaded! Dimensions: %dx%d\n", filepath, img_w, img_h);

    // [1, 3, img_h, img_w]
    Tensor raw_tensor = create_Tensor(1, 3, img_h, img_w);

    // HWC -> NCHW
    size_t plane_size = (size_t) img_h * img_w;
    for (size_t i = 0; i < plane_size; ++i) {
        raw_tensor.data[0 * plane_size + i] = (float) raw_data[i * 3 + 2]; // B
        raw_tensor.data[1 * plane_size + i] = (float) raw_data[i * 3 + 1]; // G
        raw_tensor.data[2 * plane_size + i] = (float) raw_data[i * 3 + 0]; // R
    }
    stbi_image_free(raw_data);

    // if it has good size
    if ((uint32_t) img_w == target_w && (uint32_t) img_h == target_h) {
        return raw_tensor;
    }

    //resize if not
    printf("[*] Resizing tensor from [%dx%d] to [%ux%u]...\n", img_w, img_h, target_w, target_h);
    const Tensor resized = resize_nn_asymmetric(&raw_tensor, target_w, target_h);
    free_tensor(&raw_tensor);

    return resized;
}

static void l2_normalize(float *vec, const size_t n) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_sq += vec[i] * vec[i];
    }
    const float inv_norm = 1.0f / sqrtf(sum_sq + 1e-12f);
    for (size_t i = 0; i < n; ++i) {
        vec[i] *= inv_norm;
    }
}

static float cosine_similarity(const float *a, const float *b, const size_t n) {
    float dot = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        dot += a[i] * b[i];
    }
    return dot;
}
static bool extract_face_feature(const Network *sface_net, const Tensor *aligned_face, float out_vec[128]) {
    Tensor input_face = create_Tensor(aligned_face->batch, aligned_face->channel, aligned_face->height, aligned_face->width);
    const size_t bytes = (size_t)aligned_face->batch * aligned_face->channel * aligned_face->height * aligned_face->width * sizeof(float);
    memcpy(input_face.data, aligned_face->data, bytes);

    NetworkOutputs sface_outs = forward_network(sface_net, &input_face);
    const Tensor *feat_tensor = find_output_tensor(&sface_outs, "fc1");

    if (!feat_tensor || (feat_tensor->channel * feat_tensor->batch != 128)) {
        fprintf(stderr, "[-] ERROR: Nie znaleziono tensora wyjsciowego 'fc1' (128D) w SFace!\n");
        free_network_outputs(&sface_outs);
        return false;
    }

    memcpy(out_vec, feat_tensor->data, 128 * sizeof(float));
    l2_normalize(out_vec, 128);

    free_network_outputs(&sface_outs);
    return true;
}
// input picture -> network forward (yunet)-> decoding -> nms -> face alignment
#define SFACE_COSINE_THRESHOLD 0.363f

#include "face_engine.h"

static Network g_yunet;
static Network g_sface;
static bool g_initialized = false;

int init_engine(const char *yunet_cfg, const char *yunet_weights,
                const char *sface_cfg, const char *sface_weights,
                int num_threads) {
    if (g_initialized) {
        return 0;
    }

    const int32_t total_cores = omp_get_num_procs();
    int32_t threads = (num_threads > 0) ? num_threads : (total_cores > 1 ? total_cores - 1 : 1);
    omp_set_num_threads(threads);

    g_yunet = load_network(yunet_cfg, yunet_weights);
    g_sface = load_network(sface_cfg, sface_weights);

    if (g_yunet.num_layers == 0 || g_sface.num_layers == 0) {
        return -1;
    }

    g_initialized = true;
    return 0;
}

CInferenceResult process_image_memory(const unsigned char *img_bytes, int byte_len,
                                      float score_thresh, float nms_thresh) {
    CInferenceResult result = {0, 0, NULL};

    if (!g_initialized) {
        result.error_code = -1;
        return result;
    }

    int orig_w = 0, orig_h = 0, channels = 0;
    // Decode image directly from memory buffer
    unsigned char *raw_data = stbi_load_from_memory(img_bytes, byte_len, &orig_w, &orig_h, &channels, 3);
    if (!raw_data || orig_w <= 0 || orig_h <= 0) {
        result.error_code = -2;
        return result;
    }

    // 1. Prepare and resize input tensor to [1, 3, 640, 640] for YuNet
    Tensor raw_tensor = create_Tensor(1, 3, orig_h, orig_w);
    const size_t plane_size = (size_t)orig_h * orig_w;
    for (size_t i = 0; i < plane_size; ++i) {
        raw_tensor.data[0 * plane_size + i] = (float)raw_data[i * 3 + 2]; // B
        raw_tensor.data[1 * plane_size + i] = (float)raw_data[i * 3 + 1]; // G
        raw_tensor.data[2 * plane_size + i] = (float)raw_data[i * 3 + 0]; // R
    }

    Tensor yunet_input;
    if (orig_w == 640 && orig_h == 640) {
        yunet_input = raw_tensor;
    } else {
        yunet_input = resize_nn_asymmetric(&raw_tensor, 640, 640);
        free_tensor(&raw_tensor);
    }

    // 2. Detect faces
    NetworkOutputs yunet_outs = forward_network(&g_yunet, &yunet_input);
    FaceVector faces;
    face_vec_init(&faces, 16);
    decode_yunet(&yunet_outs, &faces, 640, 640, score_thresh);
    nms(&faces, nms_thresh);

    free_tensor(&yunet_input);
    free_network_outputs(&yunet_outs);

    if (faces.size == 0) {
        stbi_image_free(raw_data);
        face_vec_free(&faces);
        return result; // count = 0
    }

    result.faces = (CDetectedFace *)calloc(faces.size, sizeof(CDetectedFace));
    if (!result.faces) {
        stbi_image_free(raw_data);
        face_vec_free(&faces);
        result.error_code = -3;
        return result;
    }

    const float scale_x = (float)orig_w / 640.0f;
    const float scale_y = (float)orig_h / 640.0f;
    int valid_count = 0;

    // 3. Align face and extract features (SFace) for each detected face
    for (size_t i = 0; i < faces.size; ++i) {
        const Face *f = &faces.faces[i];

        float landmarks[10];
        for (int k = 0; k < 5; ++k) {
            landmarks[k * 2 + 0] = f->landmarks[k * 2 + 0] * scale_x;
            landmarks[k * 2 + 1] = f->landmarks[k * 2 + 1] * scale_y;
        }

        Tensor aligned_face = align_crop_face(raw_data, orig_w, orig_h, landmarks);
        float emb[128];
        bool ok = extract_face_feature(&g_sface, &aligned_face, emb);
        free_tensor(&aligned_face);

        if (!ok) {
            continue;
        }

        // Bounding box in original image pixels: [top, right, bottom, left]
        int left   = (int)roundf(f->x * scale_x);
        int top    = (int)roundf(f->y * scale_y);
        int right  = (int)roundf((f->x + f->w) * scale_x);
        int bottom = (int)roundf((f->y + f->h) * scale_y);

        // Clamp coordinates to image boundaries
        left   = (left < 0) ? 0 : (left >= orig_w ? orig_w - 1 : left);
        top    = (top < 0) ? 0 : (top >= orig_h ? orig_h - 1 : top);
        right  = (right <= left) ? left + 1 : (right > orig_w ? orig_w : right);
        bottom = (bottom <= top) ? top + 1 : (bottom > orig_h ? orig_h : bottom);

        CDetectedFace *out_face = &result.faces[valid_count];
        out_face->top = top;
        out_face->right = right;
        out_face->bottom = bottom;
        out_face->left = left;
        out_face->confidence = f->confidence;
        memcpy(out_face->embedding, emb, 128 * sizeof(float));

        valid_count++;
    }

    result.count = valid_count;
    stbi_image_free(raw_data);
    face_vec_free(&faces);
    return result;
}

void free_inference_result(CInferenceResult *res) {
    if (res && res->faces) {
        free(res->faces);
        res->faces = NULL;
    }
    if (res) {
        res->count = 0;
    }
}

void cleanup_engine(void) {
    if (g_initialized) {
        free_network(&g_yunet);
        free_network(&g_sface);
        g_initialized = false;
    }
}
