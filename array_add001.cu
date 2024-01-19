#include<stdio.h>
#include "nvToolsExt.h"

void add_host(int* array_a, int* array_b, int* array_c, int size)
{
    nvtxRangePushA("add_host");
    for(int i = 0; i < size; ++i)
    {
        array_c[i] = array_a[i] + array_b[i];
    }
    nvtxRangePop();
}

__global__ void add_device(int* array_a, int* array_b, int* array_c)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    array_c[index] = array_a[index] + array_b[index];
}

int main()
{
    int elements = 1 << 28;

    nvtxRangePushA("host_allocation");
    int* host_a = (int*) malloc(sizeof(int) * elements);
    int* host_b = (int*) malloc(sizeof(int) * elements);
    int* host_c = (int*) malloc(sizeof(int) * elements);

    for(int i = 0; i < elements; ++i)
    {
        host_a[i] = i;
        host_b[i] = i;
    }
    nvtxRangePop();    

    // Host code

    add_host(host_a, host_b, host_c, elements);

    for(int i = 0; i < 10; ++i)
    {
        printf("Host: %d + %d = %d\n", host_a[i], host_b[i], host_c[i]);
    }
    memset(host_c, 0, sizeof(int) * elements);

    // Device code

    int* device_a;
    int* device_b;
    int* device_c;

    cudaMalloc(&device_a, sizeof(int) * elements);
    cudaMalloc(&device_b, sizeof(int) * elements);
    cudaMalloc(&device_c, sizeof(int) * elements);

    cudaMemcpy(device_a, host_a, sizeof(int) * elements, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, sizeof(int) * elements, cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid(elements / block.x);

    add_device<<<grid, block>>>(device_a, device_b, device_c);

    cudaMemcpy(host_c, device_c, sizeof(int) * elements, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; ++i)
    {
        printf("Device: %d + %d = %d\n", host_a[i], host_b[i], host_c[i]);
    }
    memset(host_c, 0, sizeof(int) * elements);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(host_a);
    free(host_b);
    free(host_c);

    // Pinned memory

    nvtxRangePushA("host_allocation");
    cudaMallocHost(&host_a, sizeof(int) * elements);
    cudaMallocHost(&host_b, sizeof(int) * elements);
    cudaMallocHost(&host_c, sizeof(int) * elements);

    for(int i = 0; i < elements; ++i)
    {
        host_a[i] = i;
        host_b[i] = i;
    }
    nvtxRangePop(); 

    cudaMalloc(&device_a, sizeof(int) * elements);
    cudaMalloc(&device_b, sizeof(int) * elements);
    cudaMalloc(&device_c, sizeof(int) * elements);

    cudaMemcpy(device_a, host_a, sizeof(int) * elements, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, sizeof(int) * elements, cudaMemcpyHostToDevice);

    add_device<<<grid, block>>>(device_a, device_b, device_c);

    cudaMemcpy(host_c, device_c, sizeof(int) * elements, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; ++i)
    {
        printf("Pinned: %d + %d = %d\n", host_a[i], host_b[i], host_c[i]);
    }
    memset(host_c, 0, sizeof(int) * elements);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);

    return 0;
}