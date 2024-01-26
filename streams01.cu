#include <stdio.h>

__global__ void omg_gpu(float* array) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < 25; ++i)
        array[index] = 1.5 * array[index] * (1 - array[index]);
}

int main() {
    int elements = 1 << 28;

    float* host;
    cudaMallocHost(&host, sizeof(float) * elements);
    for(int i = 0; i < elements; ++i) {
        host[i] = (1.f * i) / elements;
    }

    float* device;
    cudaMalloc(&device, sizeof(float) * elements);

    dim3 block(128);
    dim3 grid(elements / block.x);

    cudaMemcpy(device, host, sizeof(float) * elements, cudaMemcpyDefault);

    omg_gpu<<<grid, block>>>(device);

    cudaMemcpy(host, device, sizeof(float) * elements, cudaMemcpyDefault);

    cudaFree(device);
    cudaFreeHost(host);

    return 0;
}