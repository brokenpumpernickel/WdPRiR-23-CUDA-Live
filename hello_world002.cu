#include<stdio.h>

__global__ void hello_world()
{
    printf("Hello CUDA World - (%d, %d)\n", blockIdx.x, threadIdx.x);
}

int main()
{
    hello_world<<<3,4>>>();
    return 0;
}