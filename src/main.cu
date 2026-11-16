#include <iostream>
#include <cuda_runtime.h>

__global__ void hello_cuda()
{
    printf("Hello from GPU! Block %d, Thread %d\n",
           blockIdx.x, threadIdx.x);
}

int main()
{
    hello_cuda<<<1, 10>>>();

    cudaDeviceSynchronize();

    return 0;
}