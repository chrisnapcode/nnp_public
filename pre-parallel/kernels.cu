/* kernels.cu
 *
 *  Authors: Chris Napolin, Eric Jackson
 *
 *  CUDA kernels for train_model().
 *  Weight matrices are row-major: W1[i*H1+j], W2[i*H2+j], W3[j*CLASSES+k].
*/

#include <cuda.h>
#include <math.h>
#include "config.h"
#include "kernels.h"

__global__ void forward_layer1(float *x, float *W1, float *b1, float *h1, float *h1a) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H1) return;
    float sum = b1[j];
    for (int i = 0; i < SIZE; i++) sum += x[i] * W1[i * H1 + j];
    h1[j] = sum;
    h1a[j] = sum > 0.0f ? sum : 0.0f;
}

__global__ void forward_layer2(float *h1a, float *W2, float *b2, float *h2, float *h2a) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H2) return;
    float sum = b2[j];
    for (int i = 0; i < H1; i++) sum += h1a[i] * W2[i * H2 + j];
    h2[j] = sum;
    h2a[j] = sum > 0.0f ? sum : 0.0f;
}

__global__ void forward_output(float *h2a, float *W3, float *b3, float *out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= CLASSES) return;
    float sum = b3[k];
    for (int j = 0; j < H2; j++) sum += h2a[j] * W3[j * CLASSES + k];
    out[k] = sum;
}

__global__ void softmax_kernel(float *out, float *outa) {
    float maxv = out[0];
    for (int k = 1; k < CLASSES; k++) if (out[k] > maxv) maxv = out[k];
    float sum = 0.0f;
    for (int k = 0; k < CLASSES; k++) { outa[k] = expf(out[k] - maxv); sum += outa[k]; }
    for (int k = 0; k < CLASSES; k++) outa[k] /= sum;
}

__global__ void loss_add(float *outa, float *label, float *d_loss) {
    float l = 0.0f;
    for (int k = 0; k < CLASSES; k++) l -= label[k] * logf(outa[k] + 1e-8f);
    *d_loss += l;
}

__global__ void delta3(float *label, float *outa, float *d3) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= CLASSES) return;
    d3[k] = label[k] - outa[k];
}

__global__ void delta2(float *d3, float *W3, float *h2a, float *d2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H2) return;
    float err = 0.0f;
    for (int k = 0; k < CLASSES; k++) err += d3[k] * W3[j * CLASSES + k];
    d2[j] = err * (h2a[j] > 0.0f ? 1.0f : 0.0f);
}

__global__ void delta1(float *d2, float *W2, float *h1a, float *d1) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H1) return;
    float err = 0.0f;
    for (int k = 0; k < H2; k++) err += d2[k] * W2[j * H2 + k];
    d1[j] = err * (h1a[j] > 0.0f ? 1.0f : 0.0f);
}

__global__ void update_W3(float *W3, float *d3, float *h2a) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H2 || k >= CLASSES) return;
    W3[j * CLASSES + k] += LR * d3[k] * h2a[j];
}

__global__ void update_b3(float *b3, float *d3) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= CLASSES) return;
    b3[k] += LR * d3[k];
}

__global__ void update_W2(float *W2, float *d2, float *h1a) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H1 || j >= H2) return;
    W2[i * H2 + j] += LR * d2[j] * h1a[i];
}

__global__ void update_b2(float *b2, float *d2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H2) return;
    b2[j] += LR * d2[j];
}

__global__ void update_W1(float *W1, float *d1, float *x) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= SIZE || j >= H1) return;
    W1[i * H1 + j] += LR * d1[j] * x[i];
}

__global__ void update_b1(float *b1, float *d1) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= H1) return;
    b1[j] += LR * d1[j];
}
