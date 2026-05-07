/* kernels.h
 *
 *  Authors: Chris Napolin, Eric Jackson
 *
 *  CUDA kernel prototypes for train_model().
*/

#ifndef KERNELS_H
#define KERNELS_H

__global__ void forward_layer1(float *x, float *W1, float *b1, float *h1, float *h1a);
__global__ void forward_layer2(float *h1a, float *W2, float *b2, float *h2, float *h2a);
__global__ void forward_output(float *h2a, float *W3, float *b3, float *out);
__global__ void softmax_kernel(float *out, float *outa);
__global__ void loss_add(float *outa, float *label, float *d_loss);

__global__ void delta3(float *label, float *outa, float *d3);
__global__ void delta2(float *d3, float *W3, float *h2a, float *d2);
__global__ void delta1(float *d2, float *W2, float *h1a, float *d1);

__global__ void update_W3(float *W3, float *d3, float *h2a);
__global__ void update_b3(float *b3, float *d3);
__global__ void update_W2(float *W2, float *d2, float *h1a);
__global__ void update_b2(float *b2, float *d2);
__global__ void update_W1(float *W1, float *d1, float *x);
__global__ void update_b1(float *b1, float *d1);

#endif
