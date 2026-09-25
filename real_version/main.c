#define _POSIX_C_SOURCE 200809L
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h> //using it now but might implement my own in near future
// =======> MEMORY ALLOCATION <=======
/*
 * The most important part is not for Windows cause edge devices are mainly arm os
 * Aligment is optimalization for ARM NEON registers, ARM NEON on rasberry pi 4b is 16B
 * so 4 float numbers, most optimized will be to work on few registers simultanisly, so aligment
 * to 32B will be good to work on 4 floats at the same time with one cache read
 */
void* allocate_aligned(size_t alignment, size_t size) {
#if defined(_WIN32) || defined(_WIN64)
    return _aligned_malloc(size,alignment);
#else
    void* ptr = NULL;
    res = posix_memalign(&ptr, alignment, size);
    if (res != 0) {
        fprintf(stderr, "ERROR: posix_memalign returned code %d\n", res);
        return NULL;
    }
    return ptr;
#endif
}

void free_alligned(void* ptr) {
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
    float* data;
    size_t capacity;
    size_t size;
} Float_vector;

void vec_init(Float_vector *v, size_t init_capacity ) {
    v->size = 0;
    v->capacity = (init_capacity>0) ? init_capacity :4;
    v->data = (float*)malloc(v->capacity*sizeof(float));
    if (v->data == NULL ) {
        fprintf(stderr, "ERROR: WRONG VECTOR ALLOCATION\n");
        exit(1);
    }
}

void vec_push(Float_vector* v, float val) {
    if (v->size == v->capacity) {
        v->capacity*=2;
        float* new_data = (float*)(realloc(v->data, v->capacity * sizeof(float)));
        if (new_data == NULL) {
            fprintf(stderr, "ERROR: WRONG VECTOR REALLOCATION\n");
            exit(1);
        }
        v->data=new_data;
    }
    v->data[v->size]=val;
    ++v->size;
}

void vec_free(Float_vector* v) {
    if (v->data != NULL) {
        free(v->data);
        v->data = NULL;
    }
    v->size=0;
    v->capacity=0;
}
// ===================================

// ============> TENSOR <=============
/*
* tensor shape is [n,c,h,w] (batch size,  number of channels, height, width)
* we want it to be linear for memory so caching would be faster ( no 4 dimensions or more...)
*/
typedef struct {

    int batch,channel,height,width;
    float* data;
}Tensor;

//calculate index of pixel in location [n,c,h,w] (linear distrubiution in memory :) )
int get_index_of_pixel(Tensor t, int n, int c, int h, int w) {
    return n*(t.channel * t.width *t.height) + c*(t.width*t.height) + h*t.width +w;
}

Tensor create_Tensor(const int n, const int c, const int h, const int w) {
    Tensor t;
    t.batch= n;
    t.channel = c;
    t.height = h;
    t.width= w;

    //initializie data as zeros
    size_t numOfPixels = (size_t)(n)*c*h*w;
    size_t bytes = numOfPixels*sizeof(float);
    void * ptr = NULL;
    ptr = allocate_aligned(32,bytes);
    if (ptr == NULL) {
        fprintf(stderr, "CRITICAL ERROR: no memory for the tensor :( not good\n");
        exit(1);
    }
    t.data = (float*)(ptr);
    memset(t.data,0,bytes);
    return t;
}
// ===============================

// ===========> RELU <============
// max(0, x)
void relu(Tensor *t) {

}
// ===============================

// =========> SIGMOID <===========
// 1 / (1 + e^(-x))
void sigmoid(Tensor *t) {

}
// ===============================

int main() {
    Tensor data;
}