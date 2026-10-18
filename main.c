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
// max(0, x)
void relu(const Tensor *t) {
    const size_t total_elements = (size_t) t->batch * t->channel * t->height * t->width;
    for (size_t i = 0; i < total_elements; ++i) {
        if (t->data[i] < 0.0f) {
            t->data[i] = 0.0f;
        }
    }
}

// ===============================

// =========> SIGMOID <===========
// 1 / (1 + e^(-x))
void sigmoid(const Tensor *t) {
    const size_t total_elements = (size_t) t->batch * t->channel * t->height * t->width;
    for (size_t i = 0; i < total_elements; ++i) {
        t->data[i] = 1.0f / (1.0f + expf(-t->data[i]));
    }
}

// ===============================

// =========> RESHAPE <===========
void reshape(Tensor *t, const uint32_t new_n, const uint32_t new_c, const uint32_t new_h, const uint32_t new_w) {
    const size_t total_elements = (size_t) t->batch * t->channel * t->height * t->width;
    const size_t new_total_elements = (size_t) new_n * new_c * new_h * new_w;
    if (new_total_elements != total_elements) {
        fprintf(stderr, "ERROR: Reshape dimensions do not match original size!\n");
        exit(1);
    }
    t->batch = new_n;
    t->channel = new_c;
    t->height = new_h;
    t->width = new_w;
}

// ===============================

// ========> ADD <================
void add_tensors(Tensor *t_1, const Tensor *t_2) {
    if (t_1->channel != t_2->channel || t_1->height != t_2->height || t_1->width != t_2->width) {
        fprintf(stderr, "CRITICAL ERROR: Tensors must have the same size to add them!\n");
        exit(1);
    }
    const size_t total_t1 = (size_t) t_1->channel * t_1->batch * t_1->height * t_1->width;
    for (size_t i = 0; i < total_t1; ++i) {
        t_1->data[i] += t_2->data[i];
    }
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
                                if (in_x >= 0 && in_x < (int32_t) input->width && in_y >= 0 && in_y < (int32_t) input->height) {
                                    size_t input_indx_x = get_index_of_pixel(input, b, ic, (uint32_t) in_y, (uint32_t) in_x);
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

    //indexes of input neurons
    //in yunet only neuron ADD has 2 input neurons :3
    int32_t input1_idx;
    int32_t input2_idx;

    // kernels sizes
    uint32_t kernel_h;
    uint32_t kernel_w;

    uint32_t stride;
    uint32_t padding;
    //optimization for conv - conv+relu simultaniusly
    bool apply_relu;

    // for Reshape / Resize
    uint32_t target_w;
    uint32_t target_h;

    // weights for conv layers
    Tensor weights;
    Float_vector bias;
} Layer;

typedef struct {
    Layer *layers;
    size_t num_layers;
} Network;

Tensor forward_network(Network *net, const Tensor *input) {

    Tensor *outputs = (Tensor *)malloc(net->num_layers * sizeof(Tensor));
    if (!outputs) {
        fprintf(stderr, "CRITICAL ERROR: Memory allocation failed for outputs array\n");
        exit(1);
    }

    for (size_t i = 0; i < net->num_layers; ++i) {
        Layer *layer = &net->layers[i];
        Tensor *in1 = (layer->input1_idx == -1) ? (Tensor *)input : &outputs[layer->input1_idx];

        Tensor *in2 = NULL;
        if (layer->type == LAYER_ADD && layer->input2_idx != -1) {
            in2 = &outputs[layer->input2_idx];
        }

        if (layer->type == LAYER_CONV) {
            outputs[i] = conv(in1, &layer->weights, &layer->bias, layer->stride, layer->padding, layer->apply_relu);
        }
        else if (layer->type == LAYER_DEPTHWISE_CONV) {
            outputs[i] = depthwise_conv2D(in1, &layer->weights, &layer->bias, layer->stride, layer->padding, layer->apply_relu);
        }
        else if (layer->type == LAYER_MAXPOOL) {
            outputs[i] = maxPool(in1, layer->kernel_h, layer->stride);
        }
        else if (layer->type == LAYER_RESIZE) {
            outputs[i] = resize_nn_asymmetric(in1, layer->target_w, layer->target_h);
        }
        else if (layer->type == LAYER_ADD) {
            outputs[i] = add_tensors(in1, in2);
        }
        else {
            fprintf(stderr, "CRITICAL ERROR: Unsupported layer type %d at index %zu\n", layer->type, i);
            exit(1);
        }
    }

    Tensor final_result = outputs[net->num_layers - 1];

    for (size_t i = 0; i < net->num_layers - 1; ++i) {
        free_tensor(&outputs[i]);
    }
    free(outputs);

    return final_result;
}
int main() {
}
