#ifndef FACE_ENGINE_H
#define FACE_ENGINE_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int top;
    int right;
    int bottom;
    int left;
    float confidence;
    float embedding[128];
} CDetectedFace;

typedef struct {
    int count;
    int error_code; // 0 = OK, -1 = engine not ready, -2 = JPEG/PNG decoding error
    CDetectedFace *faces;
} CInferenceResult;

// OpenMP initialization and loading the weights to RAM
int init_engine(const char *yunet_cfg, const char *yunet_weights,
                const char *sface_cfg, const char *sface_weights,
                int num_threads);

// Full inference from the image
CInferenceResult process_image_memory(const unsigned char *img_bytes, int byte_len,
                                      float score_thresh, float nms_thresh);

// Frees allocated memory by process_image_memory
void free_inference_result(CInferenceResult *res);

// Frees neural networks loaded at the start
void cleanup_engine(void);

#ifdef __cplusplus
}
#endif

#endif
