#include <cuda_runtime.h>
#include <iostream>
#include <vector>

const int TILE_WIDTH{16};

__global__ void TiledMatrixMulKernel(float* M, float* N, float* P, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y; int ty = threadIdx.y;
    int bx = blockIdx.x; int tx = threadIdx.x;

    // the output value target for the thread, P_{Row, Col} 
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    // initialize variable to aggregate result through phases.
    float Pvalue = 0;

    for (int ph{0}; ph < (Width/TILE_WIDTH); ++ph)
    {
        // load values into shared memory. each thread will load one value from M and another from N. 
        /*
            assuming a 2x2 grid, each block having 4 threads, and being the matrices multiplied M and N each of Width 4, focusing on the first block:
                -> thread_{0,0} loads M_{0,0} and N{0,0} into shared memory in phase 0. 
                -> thread_{0,1} loads M_{0,1} and N{0,1} into shared memory in phase 0. 
                -> thread_{0,0} loads M_{0,2} and N{2,0} into shared memory in phase 1. 
                -> thread_{0,1} loads M_{0,3} and N{2,1} into shared memory in phase 1. 
                -> notice that the general rule is that thread_{ty, tx} loads M_{ty, TILE_WIDTH * ph + tx} and N_{(TILE_WIDTH * ph + ty), tx}.    
        */
       Mds[ty][tx] = M[Width*Row + TILE_WIDTH*ph + tx]; 
       Nds[ty][tx] = N[(TILE_WIDTH * ph + ty)*Width + Col];
       __syncthreads(); // dependency: we want all threads to finish writing to shared memory arrays before any reads from them. 

       // with the tile loaded into shared memory, do the calculations for the current phase:
       for (int k{0}; k<TILE_WIDTH; ++k)
       {
            Pvalue += Mds[ty][k]*Nds[k][tx];
       }
       __syncthreads(); // dependency: we want all threads to wait for data to be read by all threads before overwriting it.
    }
    P[Row*Width+Col] = Pvalue;
}

// +++++++++++++++++++++++
// the naive mat mul kernel for comparison purposes:

__global__ void MatrixMulKernelSquare(float* M, float* N, float* P, int width)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if ((row < width) && (col < width))
    {
        float Pval = 0;
        for (int k{0}; k < width; ++k)
        {
            Pval += M[row*width + k]*N[width*k + col];
        }
        P[row*width+col] = Pval;
    }
}

void printOccupancyAndResources(const void* kernel, const char* name, int threadsPerBlock)
{
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int maxActiveBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, kernel, threadsPerBlock, 0);

    int activeWarps = maxActiveBlocks * (threadsPerBlock / prop.warpSize);
    int maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    float occupancy = (static_cast<float>(activeWarps) / maxWarps) * 100.0f;

    std::cout << "\n--- Resource & Occupancy Analysis: " << name << " ---\n";
    std::cout << "  Threads per Block:      " << threadsPerBlock << "\n";
    std::cout << "  Max Active Blocks/SM:   " << maxActiveBlocks << "\n";
    std::cout << "  Theoretical Occupancy:  " << occupancy << "%\n";
}

int main()
{
    const int Width = 1024; // Must be divisible by TILE_WIDTH
    const int numElements = Width * Width;
    const size_t bytes = numElements * sizeof(float);

    std::cout << "Initializing Matrices (" << Width << " x " << Width << ")...\n";

    // Allocate host memory
    std::vector<float> h_M(numElements, 1.5f);
    std::vector<float> h_N(numElements, 2.0f);
    std::vector<float> h_P_naive(numElements, 0.0f);
    std::vector<float> h_P_tiled(numElements, 0.0f);

    // Allocate device memory
    float *d_M, *d_N, *d_P_naive, *d_P_tiled;
    cudaMalloc(&d_M, bytes);
    cudaMalloc(&d_N, bytes);
    cudaMalloc(&d_P_naive, bytes);
    cudaMalloc(&d_P_tiled, bytes);

    cudaMemcpy(d_M, h_M.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N.data(), bytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);

    // CUDA Events for Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ------------------------------------------------------------------------
    // Run Naive Kernel
    // ------------------------------------------------------------------------
    // Warmup
    MatrixMulKernelSquare<<<dimGrid, dimBlock>>>(d_M, d_N, d_P_naive, Width);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    MatrixMulKernelSquare<<<dimGrid, dimBlock>>>(d_M, d_N, d_P_naive, Width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float msNaive = 0;
    cudaEventElapsedTime(&msNaive, start, stop);

    // ------------------------------------------------------------------------
    // Run Tiled Kernel
    // ------------------------------------------------------------------------
    // Warmup
    TiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P_tiled, Width);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    TiledMatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P_tiled, Width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float msTiled = 0;
    cudaEventElapsedTime(&msTiled, start, stop);

    // Copy results back for validation
    cudaMemcpy(h_P_naive.data(), d_P_naive, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_P_tiled.data(), d_P_tiled, bytes, cudaMemcpyDeviceToHost);

    // ------------------------------------------------------------------------
    // Performance & Resource Metrics Calculation
    // ------------------------------------------------------------------------
    double ops = 2.0 * std::pow(Width, 3); // 2 * N^3 operations
    double gflopsNaive = (ops / (msNaive / 1000.0)) / 1e9;
    double gflopsTiled = (ops / (msTiled / 1000.0)) / 1e9;

    // Global memory bytes transferred (3 matrices read/written)
    double globalMemoryBytes = 3.0 * bytes; 
    double gbsNaive = (globalMemoryBytes / (msNaive / 1000.0)) / 1e9;
    double gbsTiled = (globalMemoryBytes / (msTiled / 1000.0)) / 1e9;

    std::cout << "\n==================================================\n";
    std::cout << "                BENCHMARK RESULTS                 \n";
    std::cout << "==================================================\n";

    std::cout << "Naive Execution Time:   " << msNaive << " ms\n";
    std::cout << "Naive Performance:      " << gflopsNaive << " GFLOPS\n";
    std::cout << "Naive Global Bandwidth: " << gbsNaive << " GB/s\n\n";

    std::cout << "Tiled Execution Time:   " << msTiled << " ms\n";
    std::cout << "Tiled Performance:      " << gflopsTiled << " GFLOPS\n";
    std::cout << "Tiled Global Bandwidth: " << gbsTiled << " GB/s\n\n";

    std::cout << "Speedup Factor:         " << (msNaive / msTiled) << "x faster\n";

    printOccupancyAndResources((const void*)MatrixMulKernelSquare, "Naive Kernel", TILE_WIDTH * TILE_WIDTH);
    printOccupancyAndResources((const void*)TiledMatrixMulKernel, "Tiled Kernel", TILE_WIDTH * TILE_WIDTH);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P_naive);
    cudaFree(d_P_tiled);

    return 0;
}

// results:
// ==================================================
//                 BENCHMARK RESULTS                 
// ==================================================
// Naive Execution Time:   3.50182 ms
// Naive Performance:      613.247 GFLOPS
// Naive Global Bandwidth: 3.59325 GB/s

// Tiled Execution Time:   2.16602 ms
// Tiled Performance:      991.444 GFLOPS
// Tiled Global Bandwidth: 5.80924 GB/s

// Speedup Factor:         1.61671x faster

// --- Resource & Occupancy Analysis: Naive Kernel ---
//   Threads per Block:      256
//   Max Active Blocks/SM:   4
//   Theoretical Occupancy:  100%

// --- Resource & Occupancy Analysis: Tiled Kernel ---
//   Threads per Block:      256
//   Max Active Blocks/SM:   4
//   Theoretical Occupancy:  100%


