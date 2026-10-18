#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h> //using it now but might implement my own in near future
#include <stdbool.h>
// =======> MEMORY ALLOCATION <=======
/*
 * The most important part is not for Windows cause edge devices are mainly arm os
 * Alignment is optimization for ARM NEON registers, ARM NEON on raspberry pi 4b is 16B
 * so 4 float numbers, most optimized will be to work on few registers simultaneously, so alignment
 * to 32B will be good to work on 4 floats at the same time with one cache read
 */
void *allocate_aligned(size_t alignment, size_t size) {
#if defined(_WIN32) || defined(_WIN64)
    return _aligned_malloc(size, alignment);
#else
    void *ptr = NULL;
    uint32_t res = posix_memalign(&ptr, alignment, size);
    if (res != 0) {
        fprintf(stderr, "ERROR: posix_memalign returned code %d\n", res);
        return NULL;
    }
    return ptr;
#endif
}

void free_alligned(void *ptr) {
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

void vec_init(Float_vector *v, const size_t init_capacity) {
    v->size = 0;
    v->capacity = (init_capacity > 0) ? init_capacity : 4;
    v->data = (float *) malloc(v->capacity * sizeof(float));
    if (v->data == NULL) {
        fprintf(stderr, "ERROR: WRONG VECTOR ALLOCATION\n");
        exit(1);
    }
}

void vec_push(Float_vector *v, float val) {
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

void vec_free(Float_vector *v) {
    if (v->data != NULL) {
        free(v->data);
        v->data = NULL;
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

//calculate index of pixel in location [n,c,h,w] (linear distrubiution in memory :) )
size_t get_index_of_pixel(const Tensor *t, const uint32_t n, const uint32_t c, const uint32_t h, const uint32_t w) {
    return (size_t) n * (t->channel * t->width * t->height) +
           (size_t) c * (t->width * t->height) + (size_t) h * t->width + w;
}

Tensor create_Tensor(const uint32_t n, const uint32_t c, const uint32_t h, const uint32_t w) {
    Tensor t;
    t.batch = n;
    t.channel = c;
    t.height = h;
    t.width = w;

    //initializie data as zeros
    size_t numOfPixels = (size_t) (n) * c * h * w;
    size_t bytes = numOfPixels * sizeof(float);
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

void free_tensor(Tensor *t) {
    if (t->data != NULL) {
        free_alligned(t->data);
        t->data = NULL;
    }
    t->batch = t->channel = t->height = t->width = 0;
}

// ===============================

// ===========> RELU <============
// lets do this in place
// max(0, x)
Tensor relu(Tensor *t) {
    const size_t total_elements = (size_t) t->batch * t->channel * t->height * t->width;
    for (size_t i = 0; i < total_elements; ++i) {
        if (t->data[i] < 0.0f) {
            t->data[i] = 0.0f;
        }
    }
    Tensor res = *t;
    t->data = NULL;
    t->batch = t->channel = t->height = t->width = 0;
    return res;
}

// ===============================

// =========> SIGMOID <===========
// 1 / (1 + e^(-x))
Tensor sigmoid(Tensor *t) {
    const size_t total_elements = (size_t) t->batch * t->channel * t->height * t->width;
    for (size_t i = 0; i < total_elements; ++i) {
        t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
    }
    Tensor res = *t;
    t->data = NULL;
    t->batch = t->channel = t->height = t->width = 0;
    return res;
}

// ===============================

// =========> RESHAPE <===========
Tensor reshape(Tensor *t, const uint32_t new_n, const uint32_t new_c, const uint32_t new_h, const uint32_t new_w) {
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

    t->data = NULL;
    t->batch = t->channel = t->height = t->width = 0;
    return res;
}

// ===============================

// ========> ADD <================
Tensor add_tensors(Tensor *t_1, const Tensor *t_2) {
    Tensor res = create_Tensor(t_1->batch, t_1->channel, t_1->height, t_1->width);
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
Tensor transpose(const Tensor *t) {
    //                              n           c           h               w
    Tensor res = create_Tensor(t->batch, t->height, t->width, t->channel);
    for (uint32_t b = 0; b < t->batch; ++b) {
        for (uint32_t c = 0; c < t->channel; ++c) {
            for (uint32_t h = 0; h < t->height; ++h) {
                for (uint32_t w = 0; w < t->width; ++w) {
                    size_t orginial_indx = get_index_of_pixel(t, b, c, h, w);
                    size_t after_indx = (size_t) b * (t->height * t->channel * t->width) +
                                        (size_t) h * (t->channel * t->width) +
                                        (size_t) w * t->channel + c;
                    res.data[after_indx] = t->data[orginial_indx];
                }
            }
        }
    }
    return res;
}

// ===============================

// =======> MAXPOOL_2D <=============
Tensor maxPool(const Tensor *t, uint32_t pool_size, uint32_t stride) {
    uint32_t new_h = (t->height - pool_size) / stride + 1;
    uint32_t new_w = (t->width - pool_size) / stride + 1;
    Tensor res = create_Tensor(t->batch, t->channel, new_h, new_w);
    for (uint32_t b = 0; b < t->batch; ++b) {
        for (uint32_t c = 0; c < t->channel; ++c) {
            for (uint32_t h = 0; h < new_h; ++h) {
                for (uint32_t w = 0; w < new_w; ++w) {
                    uint32_t res_indx = get_index_of_pixel(&res, b, c, h, w);
                    float max_val = -INFINITY;
                    for (uint32_t i = 0; i < pool_size; ++i) {
                        for (uint32_t j = 0; j < pool_size; ++j) {
                            uint32_t t_i = (h * stride) + i; //columna ten
                            uint32_t t_j = (w * stride) + j; //row
                            uint32_t t_index = get_index_of_pixel(t, b, c, t_i, t_j);
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
Tensor resize_nn_asymmetric(const Tensor *t, const uint32_t new_width, const uint32_t new_height) {
    Tensor res = create_Tensor(t->batch, t->channel, new_height, new_width);
    float scale_x = (float) new_width / (float) t->width;
    float scale_y = (float) new_height / (float) t->height;
    for (uint32_t b = 0; b < t->batch; ++b) {
        for (uint32_t c = 0; c < t->channel; ++c) {
            for (uint32_t y = 0; y < new_height; ++y) {
                for (uint32_t x = 0; x < new_width; ++x) {
                    uint32_t old_y = (uint32_t) (y / scale_y);
                    uint32_t old_x = (uint32_t) (x / scale_x);

                    if (old_y >= t->height) old_y = t->height - 1;
                    if (old_x >= t->width) old_x = t->width - 1;

                    size_t new_tensor_index = get_index_of_pixel(&res, b, c, y, x);
                    size_t old_tensor_index = get_index_of_pixel(t, b, c, old_y, old_x);

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

// ===========> PRLEU <===========
// ===============================

// =======> Flatten <=============
// ===============================

// =========> subtract <==========
// ===============================

// ========> multiply <===========
// ===============================

// ==========> CONV <=============
//classical conv
// out size formula is same for height and width O= floor((I-K+2P)/S) + 1, where I is input size, K is kernel size,
// p is padding and S is Stride
//  filter tensor params are [C_out, C_in, K_h, K_w] - c_out is number of channels on out tensor and C_in is channels _in
Tensor conv(const Tensor *input, const Tensor *filter_tensor, const Float_vector *bias, const uint32_t stride,
            const uint32_t padding, bool apply_relu) {
    //more clean names
    uint32_t out_channels = filter_tensor->batch; // number of filters
    uint32_t in_channels = filter_tensor->channel; // filter Z coordinate
    uint32_t kernel_h = filter_tensor->height; // filter Y coordinate
    uint32_t kernel_w = filter_tensor->width; // filter X coordinate

    if (input->channel != in_channels) {
        fprintf(stderr, "CRITICAL ERROR: Input channels (%u) do not match weight in_channels (%u)!\n", input->channel,
                in_channels);
        exit(1);
    }

    uint32_t out_h = (input->height - kernel_h + 2 * padding) / stride + 1;
    uint32_t out_w = (input->width - kernel_w + 2 * padding) / stride + 1;
    Tensor out = create_Tensor(input->batch, out_channels, out_h, out_w);

    for (uint32_t b = 0; b < input->batch; ++b) {
        for (uint32_t c = 0; c < out_channels; ++c) {
            float bias_val = (bias != NULL && bias->size > c) ? bias->data[c] : 0.0f;
            for (uint32_t y = 0; y < out_h; ++y) {
                for (uint32_t x = 0; x < out_w; ++x) {
                    float sum = bias_val;
                    // we go through every filter tensor and tensor has parameters [ic,kh,kw] b in filter tensor is numvber of tensors
                    for (uint32_t ic = 0; ic < in_channels; ++ic) {
                        for (uint32_t ky = 0; ky < kernel_h; ++ky) {
                            for (uint32_t kx = 0; kx < kernel_w; ++kx) {
                                //calculate parameters of input tensor
                                int32_t in_y = (y * stride) + ky - padding;
                                int32_t in_x = (x * stride) + kx - padding;
                                //check if it is not padding (zeros)
                                if (in_x >= 0 && in_x < (int32_t) input->width && in_y >= 0 && in_y < (int32_t) input->
                                    height) {
                                    size_t input_indx_x = get_index_of_pixel(
                                        input, b, ic, (uint32_t) in_y, (uint32_t) in_x);
                                    size_t filter_tensor_x = get_index_of_pixel(filter_tensor, c, ic, ky, kx);

                                    sum += input->data[input_indx_x] * filter_tensor->data[filter_tensor_x];
                                }
                            }
                        }
                    }
                    if (apply_relu && sum < 0.0f) {
                        sum = 0.0f;
                    }

                    size_t out_idx = get_index_of_pixel(&out, b, c, y, x);
                    out.data[out_idx] = sum;
                }
            }
        }
    }
    return out;
}


// ===============================

// it is numvber of channels times faster than standard conv!
// =======> DepthWise Conv <======
Tensor depthwise_conv2D(const Tensor *input, const Tensor *filter_tensor, const Float_vector *bias,
                        const uint32_t stride,
                        const uint32_t padding, bool apply_relu) {
    // In Depthwise conv weights parameters are [number of channels, 1, kernel_h,kernel_w]
    uint32_t channels = filter_tensor->batch; // number of filters
    uint32_t kernel_h = filter_tensor->height; // filter Y coordinate
    uint32_t kernel_w = filter_tensor->width; // filter X coordinate

    if (input->channel != channels) {
        fprintf(stderr, "CRITICAL ERROR: Input channels (%u) do not match weight filters (%u)!\n", input->channel,
                channels);
        exit(1);
    }

    uint32_t out_h = (input->height - kernel_h + 2 * padding) / stride + 1;
    uint32_t out_w = (input->width - kernel_w + 2 * padding) / stride + 1;
    Tensor out = create_Tensor(input->batch, channels, out_h, out_w);

    for (uint32_t b = 0; b < input->batch; ++b) {
        for (uint32_t c = 0; c < channels; ++c) {
            float bias_val = (bias != NULL && bias->size > c) ? bias->data[c] : 0.0f;

            for (uint32_t y = 0; y < out_h; ++y) {
                for (uint32_t x = 0; x < out_w; ++x) {
                    float sum = bias_val;
                    // we go through every filter tensor (2 dimensions not 3 like standard conv [kw,kh]
                    // b in filter tensor is numvber of tensor
                    for (uint32_t ky = 0; ky < kernel_h; ++ky) {
                        for (uint32_t kx = 0; kx < kernel_w; ++kx) {
                            //calculate parameters of input tensor
                            int32_t in_y = (y * stride) + ky - padding;
                            int32_t in_x = (x * stride) + kx - padding;
                            //check if it is not padding (zeros)
                            if (in_x >= 0 && in_x < (int32_t) input->width && in_y >= 0 && in_y < (int32_t) input->
                                height) {
                                size_t input_indx_x = get_index_of_pixel(input, b, c, (uint32_t) in_y, (uint32_t) in_x);
                                size_t filter_tensor_x = get_index_of_pixel(filter_tensor, c, 0, ky, kx);

                                sum += input->data[input_indx_x] * filter_tensor->data[filter_tensor_x];
                            }
                        }
                    }
                    if (apply_relu && sum < 0.0f) {
                        sum = 0.0f;
                    }

                    size_t out_idx = get_index_of_pixel(&out, b, c, y, x);
                    out.data[out_idx] = sum;
                }
            }
        }
    }
    return out;
}

// ===============================


// ===========> NEURAL NETWORK  <============

typedef enum {
    LAYER_CONV,
    LAYER_DEPTHWISE_CONV,
    LAYER_MAXPOOL,
    LAYER_RESIZE,
    LAYER_TRANSPOSE,
    LAYER_ADD,
    LAYER_RESHAPE,
    LAYER_SIGMOID,
    LAYER_RELU
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

    // Dimensions for Reshape / Resize
    uint32_t out_n, out_c, out_h, out_w;


    // weights for conv layers
    Tensor weights;
    Float_vector bias;

    // Execution & Lifetime
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

typedef struct {
    OutputTensor *outputs;
    size_t count;
} NetworkOutputs;

Tensor *get_output(NetworkOutputs *outs, const char *name) {
    for (size_t i = 0; i < outs->count; ++i) {
        if (strcmp(outs->outputs[i].name, name) == 0) {
            return &outs->outputs[i].tensor;
        }
    }
    return NULL;
}

void free_network_outputs(NetworkOutputs *outs) {
    for (size_t i = 0; i < outs->count; ++i) {
        free_tensor(&outs->outputs[i].tensor);
    }
    if (outs->outputs != NULL) {
        free(outs->outputs);
        outs->outputs = NULL;
    }
    outs->count = 0;
}
void free_network(Network *net) {
    for (size_t i = 0; i < net->num_layers; ++i) {
        if (net->layers[i].type == LAYER_CONV || net->layers[i].type == LAYER_DEPTHWISE_CONV) {
            free_tensor(&net->layers[i].weights);
            vec_free(&net->layers[i].bias);
        }
    }
    if (net->layers) {
        free(net->layers);
        net->layers = NULL;
    }
    net->num_layers = 0;
}
Tensor load_weights(const char *filepath, uint32_t n, uint32_t c, uint32_t h, uint32_t w) {
    Tensor t = create_Tensor(n, c, h, w);
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "CRITICAL ERROR: Cannot open weight file %s\n", filepath);
        exit(1);
    }
    size_t total_floats = (size_t) n * c * h * w;
    size_t read_count = fread(t.data, sizeof(float), total_floats, f);
    if (read_count != total_floats) {
        fprintf(stderr, "ERROR: Expected %zu floats in %s, read %zu\n", total_floats, filepath, read_count);
    }
    fclose(f);
    return t;
}
void load_bias(const char *filepath, Float_vector *vec, uint32_t size) {
    vec_init(vec, size);
    FILE *f = fopen(filepath, "rb");
    if (!f) {
        fprintf(stderr, "CRITICAL ERROR: Cannot open bias file %s\n", filepath);
        exit(1);
    }
    size_t read_count = fread(vec->data, sizeof(float), size, f);
    if (read_count != size) {
        fprintf(stderr, "ERROR: Expected %u bias floats in %s, read %zu\n", size, filepath, read_count);
    }
    vec->size = size;
    fclose(f);
}

Network load_network(const char *cfg_path, const char *weights_dir) {
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
            int count = sscanf(s_buf, "%u,%u,%u,%u", &d1, &d2, &d3, &d4);
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

        if (l->type == LAYER_CONV || l->type == LAYER_DEPTHWISE_CONV) {
            char w_file[128] = {0};
            char b_file[128] = {0};
            if ((p = strstr(line, "weights="))) sscanf(p + 8, "%127s", w_file);
            if ((p = strstr(line, "bias="))) sscanf(p + 5, "%127s", b_file);

            char *ws = strstr(line, "w_shape=");
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

        idx++;
    }

    fclose(f);
    return net;
}

NetworkOutputs forward_network(Network *net, const Tensor *input) {
    //potential optimalization but for yunet it is only 2KB so it is ok i think
    Tensor *outputs = (Tensor *) malloc(net->num_layers * sizeof(Tensor));
    if (!outputs) {
        fprintf(stderr, "CRITICAL ERROR: Memory allocation failed for outputs array\n");
        exit(1);
    }
    //keep it for dynymic free
    int32_t *live_refs = (int32_t *) malloc(net->num_layers * sizeof(int32_t));
    for (size_t k = 0; k < net->num_layers; ++k) {
        live_refs[k] = net->layers[k].ref_count;
    }

    for (size_t i = 0; i < net->num_layers; ++i) {
        Layer *layer = &net->layers[i];
        //check if it is input layer
        Tensor *in1 = (layer->input1_idx == -1) ? (Tensor *) input : &outputs[layer->input1_idx];
        Tensor *in2 = (layer->input2_idx != -1) ? &outputs[layer->input2_idx] : NULL;

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


int main() {
    printf("[*] Loading network...\n");
    Network net = load_network("models/yunet.cfg", "models/yunet_weights");
    printf("[+] Network loaded successfully! Num layers: %zu\n", net.num_layers);

    printf("[*] Creating dummy input [1, 3, 640, 640]...\n");
    Tensor input = create_Tensor(1, 3, 640, 640);
    size_t total = 1 * 3 * 640 * 640;
    for (size_t i = 0; i < total; ++i) input.data[i] = 1.0f;

    printf("[*] Running forward pass...\n");
    NetworkOutputs outs = forward_network(&net, &input);
    printf("[+] Forward pass completed! Output count: %zu\n", outs.count);

    for (size_t i = 0; i < outs.count; ++i) {
        Tensor *t = &outs.outputs[i].tensor;
        printf("    -> Output [%zu]: '%s' shape: [%u, %u, %u, %u], first val: %.4f\n",
               i, outs.outputs[i].name, t->batch, t->channel, t->height, t->width, t->data[0]);
    }

    free_tensor(&input);
    free_network_outputs(&outs);
    free_network(&net);
    printf("[+] Cleaned up memory. All good!\n");
    return 0;
}
