#include <cuda_runtime.h>

const int TILE_WIDTH{16};

__global__ void TiledMatrixMulKernel(float* M, float* N, float* P, int Width)
{
    __shared__ float* Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float* Nds[TILE_WIDTH][TILE_WIDTH];

    int by{blockIdx.y}; int ty{threadIdx.y};
    int bx{blockIdx.x}; int tx{threadIdx.x};

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
    }
}
