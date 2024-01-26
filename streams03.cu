#include <stdio.h>

__global__ void omg_gpu(float* array) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = 0; i < 25; ++i)
        array[index] = 1.5 * array[index] * (1- array[index]);
}

int main() {
    int elements = 1 << 25;

    float* host;
    cudaMallocHost(&host, sizeof(float) * elements);
    for(int i = 0; i < elements; ++i) {
        host[i] = (1.f * i) / elements;
    }

    float* device;
    cudaMalloc(&device, sizeof(float) * elements);

    int nstreams = 4;
    cudaStream_t* streams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * nstreams);
    for(int i = 0; i < nstreams; ++i)
        cudaStreamCreate(&streams[i]);

    int stream_elements = elements / nstreams;
    dim3 block(128);
    dim3 grid(stream_elements / block.x);

    for(int i = 0; i < nstreams; ++i) {
        cudaMemcpyAsync(&device[i * stream_elements], &host[i * stream_elements], sizeof(float) * stream_elements, cudaMemcpyDefault, streams[i]);
    }

    for(int i = 0; i < nstreams; ++i) {
        omg_gpu<<<grid, block, 0, streams[i]>>>(&device[i * stream_elements]);
    }

    for(int i = 0; i < nstreams; ++i) {
        cudaMemcpyAsync(&host[i * stream_elements], &device[i * stream_elements], sizeof(float) * stream_elements, cudaMemcpyDefault, streams[i]);
    }

    for(int i = 0; i < nstreams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(device);
    cudaFreeHost(host);

    return 0;
}