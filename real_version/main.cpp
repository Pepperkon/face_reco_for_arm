#include <cstdlib>
#include <cstdio>
#include <wchar.h>

//Vector declaration for more flexibility
typedef struct {
    float* data;
    size_t capacity;
    size_t size;
} Float_vector;

void vec_init(Float_vector v, size_t init_capacity ) {
    v.size = 0;
    v.capacity = (init_capacity>0) ? init_capacity :4;
    v.data = (float*)malloc(v.capacity*sizeof(float));
    if (v.data == nullptr) {
        fprintf(stderr, "ERROR: WRONG VECTOR ALLOCATION\n");
        exit(1);
    }
}

void vec_push(Float_vector v, float val) {
    if (v.size == v.capacity) {
        v.capacity*=2;
        float* new_data = (float*)realloc(v.data, v.capacity*sizeof(float));
        if (new_data == nullptr) {
            fprintf(stderr, "ERROR: WRONG VECTOR REALLOCATION\n");
            exit(1);
        }
        v.data=new_data;
    }
    v.data[v.size]=val;
    ++v.size;
}

void vec_free(Float_vector* v) {
    if (v->data != nullptr) {
        free(v->data);
        v->data = nullptr;
    }
    v->size=0;
    v->capacity=0;
}

typedef struct {
    //tensor shape is [n,c,h,w] (batch size,  number of channels, height, width)
    //we want it to be linear for memory so caching would be faster
    int batch,channel,height,width;
    Float_vector* data;
}Tensor;

//calculate index of pixel in location [n,c,h,w] (linear distrubiution in memory :) )
int get_index_of_pixel(Tensor t, int n, int c, int h, int w) {
    return n*(t.channel * t.width *t.height) + c*(t.width*t.height) + h*t.width +w;
}

Tensor create_Tensor(int n, int c, int h, int w) {
    Tensor t;
    t.batch= n;;
    t.channel = c;
    t.height = h;
    t.width= w;

    //initializie data as zeros
    size_t numOfPixels = (size_t)n*c*h*w;
    size_t
    return t;
}


int main() {
    Tensor data;
}