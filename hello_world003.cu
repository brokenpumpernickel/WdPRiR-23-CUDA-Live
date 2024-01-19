#include<stdio.h>

__global__ void hello_world()
{
    printf("Hello CUDA World - (%d, %d) (%d, %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main()
{
    dim3 grid(3, 3);
    dim3 block(2, 2);
    hello_world<<<grid,block>>>();
    return 0;
}