#include <cuda_runtime.h>
#include <iostream>
#include <vector>

const int TILE_WIDTH{16};

// simple rule: if input matrices M_{j, k} and N_{k, l} are multiplied, we obtain matrix P_{j, l}

__global__ void TiledMatrixMulKernel(float* M, float* N, float* P, int j, int k, int l)
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

    // to avoid type conversion overhead (int -> float -> ceilf -> int), use pure integer arithmetic directly. 
    int numPhases = (k + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int ph{0}; ph < numPhases; ++ph)
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

        // input and output boundary guards
       if (Row < j && (TILE_WIDTH * ph + tx) < k) 
       {
            Mds[ty][tx] = M[k*Row + TILE_WIDTH*ph + tx];
       }
       else Mds[ty][tx] = 0.0f;
       if (Col < l && (TILE_WIDTH *ph + ty) < k) 
       {
            Nds[ty][tx] = N[(TILE_WIDTH * ph + ty)*l + Col];
       }
       else Nds[ty][tx] = 0.0f;
       __syncthreads(); // dependency: we want all threads to finish writing to shared memory arrays before any reads from them. 

       // with the tile loaded into shared memory, do the calculations for the current phase:
       for (int i{0}; i<TILE_WIDTH; ++i)
       {
            Pvalue += Mds[ty][i]*Nds[i][tx];
       }
       __syncthreads(); // dependency: we want all threads to wait for data to be read by all threads before overwriting it.
    }

    // output boundary guard: only allow threads writing a valid P value 
    if (Row < j && Col < l)
        P[Row*l+Col] = Pvalue;
}

int main()
{
    // Small non-square dimensions: M is (18x20), N is (20x5), P is (18x5)
    int j = 18, k = 20, l = 5;

    // Fill inputs with 1.0f -> Every output element should equal k (20.0f)
    std::vector<float> h_M(j * k, 1.0f);
    std::vector<float> h_N(k * l, 1.0f);
    std::vector<float> h_P(j * l, 0.0f);

    float *d_M, *d_N, *d_P;
    cudaMalloc(&d_M, j * k * sizeof(float));
    cudaMalloc(&d_N, k * l * sizeof(float));
    cudaMalloc(&d_P, j * l * sizeof(float));

    cudaMemcpy(d_M, h_M.data(), j * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N.data(), k * l * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((l + TILE_WIDTH - 1) / TILE_WIDTH, 
                (j + TILE_WIDTH - 1) / TILE_WIDTH);

    TiledMatrixMulKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, j, k, l);
    cudaDeviceSynchronize();

    cudaMemcpy(h_P.data(), d_P, j * l * sizeof(float), cudaMemcpyDeviceToHost);

    // Verification
    bool success = true;
    for (int i = 0; i < j * l; ++i) {
        if (h_P[i] != static_cast<float>(k)) {
            std::cout << "Error at index " << i << ": expected " << k << ", got " << h_P[i] << "\n";
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "SUCCESS! All " << (j * l) << " elements in P strictly equal " << k << ".\n";
    }

    cudaFree(d_M); cudaFree(d_N); cudaFree(d_P);
    return 0;
}
