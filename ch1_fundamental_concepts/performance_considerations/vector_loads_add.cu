#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// as presented in textbook, no changes.
__global__ void vecadd_kernel_(float* x, float* y, float*z, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // if (i < n)
    //    z[i] = x[i] + y[i]
    // -> under this approach, every 4 bytes accessed issues a load/store instruction. 
    //    ideally, we want each thread to access as many consecutive bytes as possible in a single instruction. 
    float4 x4 = ((float4*)x)[i];
    float4 y4 = ((float4*)y)[i];
    float4 z4;
    // -> float4 is a CUDA built-in struct containing 4 single-precision floats named .x, .y, .z and .w, taking 16 bytes of contigous memory. 
    //    indexing on a float4* is based on a pointer arithmetic that jumps 16-byte blocks, rather than 4-byte blocks. 
    //    so thread 0 will read bytes 0-15 from x4 and z4, thread 1 will read bytes 16-31, etc. 

    z4.x = x4.x + y4.x;
    z4.y = x4.y + y4.y;
    z4.z = x4.z + y4.z;
    z4.w = x4.w + y4.w;
    ((float4*)z)[i] = z4;
}

// same as above, but with boundary checks. 
__global__ void vecadd_kernel(float* x, float* y, float* z, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = i*4; // starting scalar index. for instance, if i = 1, then the starting element will be 4 (byte 16 onwards, to 31).
    if (idx >= N)
    {
        return;
    }

    if (idx + 3 < N)
    {
    // then do as usual
        float4 x4 = ((float4*)x)[i];
        float4 y4 = ((float4*)y)[i];
        float4 z4;
        // -> float4 is a CUDA built-in struct containing 4 single-precision floats named .x, .y, .z and .w, taking 16 bytes of contigous memory. 
        //    indexing on a float4* is based on a pointer arithmetic that jumps 16-byte blocks, rather than 4-byte blocks. 
        //    so thread 0 will read bytes 0-15 from x4 and z4, thread 1 will read bytes 16-31, etc. 

        z4.x = x4.x + y4.x;
        z4.y = x4.y + y4.y;
        z4.z = x4.z + y4.z;
        z4.w = z4.w + y4.w;
        ((float4*)z)[i] = z4;
    }
    else
    {
        for (int j = idx; j < N; ++j)
        {
            z[j] = x[j] + y[j];
        }
    }

    // otherwise, we have some remaining elements that cant be put in a float4 struct.     
}

int main()
{
    // now the kernel works with vector whose length N is not a multiple of 4:
    const int N = 1000003;
    const size_t bytes = N * sizeof(float);

    std::cout << "Testing Vec Addition with N = " << N << " elements..." << std::endl;

    std::vector<float> x_h(N);
    std::vector<float> y_h(N);
    std::vector<float> z_h(N);

    for (int i{0}; i < N; ++i)
    {
        x_h[i] = static_cast<float>(i);
        y_h[i] = static_cast<float>(i);
    }

    float* x_d = nullptr;
    float* y_d = nullptr;
    float* z_d = nullptr;
    CUDA_CHECK(cudaMalloc(&x_d, bytes));
    CUDA_CHECK(cudaMalloc(&y_d, bytes));
    CUDA_CHECK(cudaMalloc(&z_d, bytes));

    CUDA_CHECK(cudaMemcpy(x_d, x_h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_d, y_h.data(), bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int numVecElements = (N+3)/4;
    int blocksPerGrid = (numVecElements + threadsPerBlock - 1)/threadsPerBlock;

    vecadd_kernel<<<blocksPerGrid, threadsPerBlock>>>(x_d, y_d, z_d, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());    

    CUDA_CHECK(cudaMemcpy(z_h.data(), z_d, bytes, cudaMemcpyDeviceToHost));

// Specific inspection of the tail elements
    std::cout << "\n--- Tail Elements Inspection ---" << std::endl;
    for (int i = N - 4; i < N; ++i) {
        std::cout << "Index " << i << " -> x: " << x_h[i] 
                  << ", y: " << y_h[i] << ", z: " << x_h[i] << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
    CUDA_CHECK(cudaFree(z_d));

    return 0;
}

