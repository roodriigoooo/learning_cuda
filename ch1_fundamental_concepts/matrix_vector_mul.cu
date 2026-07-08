#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <random>

__global__ void MatrixVectorMulKernel(float* B, float* c, float* a, int width)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < width)
    {
        float a_val{0};
        for (int k{0}; k < width; ++k)
        {
            a_val += B[row*width+k]*c[k];
        }
        a[row] = a_val;
    }
}


int main()
{
    const int width{512};
    const int matrixSize{width*width};
    const int vecSize{width};
    const int matrixBytes = matrixSize*sizeof(float);
    const int vecBytes = vecSize*sizeof(float);

    std::vector<float> B_h(matrixSize);
    std::vector<float> c_h(vecSize);
    std::vector<float> a_h(vecSize, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i{0}; i < matrixSize; ++i)
    {
        B_h[i] = dis(gen);
    }

    for (size_t i{0}; i < vecSize; ++i)
    {
        c_h[i] = dis(gen);
    }

    float* B_d = nullptr;
    float* c_d = nullptr;
    float* a_d = nullptr;

    cudaMalloc((void**)&B_d, matrixBytes);
    cudaMalloc((void**)&c_d, vecBytes);
    cudaMalloc((void**)&a_d, vecBytes);
    
    cudaMemcpy(B_d, B_h.data(), matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c_h.data(), vecBytes, cudaMemcpyHostToDevice);

    dim3 dimGrid(static_cast<int>(std::ceil(static_cast<float>(width) / 64.0f)), 1, 1);
    dim3 dimBlock(64, 1, 1);

    MatrixVectorMulKernel<<<dimGrid, dimBlock>>>(B_d, c_d, a_d, width);

    cudaMemcpy(a_h.data(), a_d, vecBytes, cudaMemcpyDeviceToHost);

    cudaFree(B_d);
    cudaFree(c_d);
    cudaFree(a_d);

    std::cout << "Kernel completed successfully! First element: " << a_h[0] << std::endl;

    return 0;
}
