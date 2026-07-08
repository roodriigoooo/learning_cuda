#include <iostream>
#include <cmath>
#include <random>
#include <iomanip>
#include <vector>
#include <cuda_runtime.h>

//assuming symmetric, square matrices.
//M and N have dimensions width x width, and so does P, the resulting matrix.
__global__ void MatrixMulKernelSquare(float* M, float* N, float* P, int width)
{
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  if ((row < width) && (col < width))
  {
    float Pval{0};
    for (int k{0}; k < width; ++k)
    {
      Pval += M[row*width+k]*N[k*width + col];
    }
    P[row*width + col] = Pval;
  }
}

//write a kernel that has each thread produce one output matrix row. 
//Fill in the execution configuration parameters for the design. 
__global__ void MatrixMulKernelSquareRow(float* M, float* N, float* P, int width)
{
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  if (row < width)
  {
    for (int c{0}; c < width; ++c)
    {
      float Pval{0};
      for (int k{0}; k < width;  ++k)
      {
        Pval += M[row*width+k]*N[width*k+c];
      }
      P[row*width + c] = Pval;
    }
  }
}

//b. Write a kernel that has each thread to produce one output matrix column. 
// Fill in the execution configuration parameters for the design.
__global__ void MatrixMulKernelSquareColumn(float* M, float* N, float* P, int width)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  if (col < width)
  {
    for (int r{0}; r < width; ++r)
    {
      float Pval{0};
      for (int k{0}; k < width; ++k)
      {
        Pval += M[r*width+k]*N[k*width+col];
      }
      P[r*width+col] = Pval;
    }
  }
}


int main()
{
    const int width{512};
    const int matrixSize{width*width};
    const int bytes = matrixSize*sizeof(float);

    std::vector<float> M_h(matrixSize);
    std::vector<float> N_h(matrixSize);
    std::vector<float> P_h(matrixSize, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (size_t i{0}; i < matrixSize; ++i)
    {
        M_h[i] = dis(gen);
        N_h[i] = dis(gen);
    }

    // allocate device pointers
    float* M_d = nullptr;
    float* N_d = nullptr;
    float* P_d = nullptr;

    cudaMalloc((void**)&M_d, bytes);
    cudaMalloc((void**)&N_d, bytes);
    cudaMalloc((void**)&P_d, bytes);

    // transfer necessary data to device for calculations

    dim3 dimGrid(static_cast<int>(std::ceil(static_cast<float>(width)/16.0f)),
                 static_cast<int>(std::ceil(static_cast<float>(width)/16.0f)),
                 1);
    dim3 dimBlock(16, 16, 1);

    cudaMemcpy(M_d, M_h.data(),bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h.data(),bytes, cudaMemcpyHostToDevice);
    MatrixMulKernelSquare<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, width);

    cudaMemcpy(P_h.data(), P_d, bytes, cudaMemcpyDeviceToHost);

    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    for (auto element: P_h)
      std::cout << element << ' ';
    std::cout << '\n';

    return 0;
}
