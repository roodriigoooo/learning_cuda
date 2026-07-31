#include <cuda_runtime.h>
#include <iostream>
#include <vector>

const int TILE_WIDTH{16};

// simple rule: if input matrices M_{j, k} and N_{k, l} are multiplied, we obtain matrix P_{j, l}
// we now assume that N is stored in column-major layout, and our task is to apply corner tuning to make accesses to the N structure coalesced. 
// -> we therefore do not want a single thread to access elements in the same column. elements in the same column will be stored contiguously in memory, and hence we want 
// -> thread_{0,0} to access N[0], thread_{0,1} to access N[1], and so on, where N[0] and N[1] now represent N_{0,0} and N{1, 0} in the N_{row, col} format. 

__global__ void TiledMatrixMulWithCornerTurningKernel(float* M, float* N, float* P, int j, int k, int l)
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
    int numPhases = k + TILE_WIDTH - 1 / TILE_WIDTH;
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
