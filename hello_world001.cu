#include<stdio.h>

__global__ void hello_world()
{
    printf("Hello CUDA World\n");
}

int main()
{
    hello_world<<<1,12>>>();
    return 0;
}