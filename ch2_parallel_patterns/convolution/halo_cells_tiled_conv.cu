#include <cuda_runtime.h>

#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM - 2*(FILTER_RADIUS)))

/*
here, we implement tiled convolution with halo cells, following a block organization such that each thread block has the dimensions of the input tile, 
and where therefore each thread loads one element onto shared memory from the input tile, and where we need to disable some threads when calculating output elements. 
the alternative approach (having each thread block have the dimensions of the smaller output tile instead) is also implemented below, in this same file.
*/

__constant__ float F_c[2*FILTER_RADIUS + 1][2*FILTER_RADIUS+1];

__global__ void conv_tiled_2d_const_mem_kernel_in(float* N, float* P, int width, int height)
{
    // this here loads the INPUT tile element responsible for loading. 
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (col < width && col >= 0 && row < height && row >= 0)
    {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    }
    else
    {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    // true dependency here. we want all threads to finish writing before any of them reads shared memory. 
    __syncthreads();

    // the input loading part is finished here. now we do the output value calculations.
    int tileCol = (int)threadIdx.x - FILTER_RADIUS; // avoid unintended wrapping
    int tileRow = (int)threadIdx.y - FILTER_RADIUS;
    
    if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM && col < width && row < height)
    {
        float Pval = 0.0f;
        #pragma unroll
        for (int fCol{0}; fCol < 2*FILTER_RADIUS + 1; ++fCol)
        {
            #pragma unroll
            for (int fRow{0}; fRow < 2*FILTER_RADIUS+1; ++fRow)
            {
                Pval += F_c[fRow][fCol]*N_s[tileRow + fRow][tileCol + fCol];
            }
        }
        P[row * width + col] = Pval;
    }
}

/*
alternative approach:
*/
__global__ void conv_tiled_2d_const_mem_kernel_out(float* N, float* P, int width, int height)
{
    const int startCol = blockIdx.x * OUT_TILE_DIM - FILTER_RADIUS;
    const int startRow = blockIdx.y * OUT_TILE_DIM - FILTER_RADIUS;


    const int threadsPerBlock = OUT_TILE_DIM * OUT_TILE_DIM;

    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    /*
    we have threadsPerBlock < IN_TILE_DIM * IN_TILE_DIM. 
    assuming IN_TILE_DIM = 32 and OUT_TILE_DIM = 28 (R = 2), we will have 784 threads in the block, yet be forced to load 1024 elements into shared memory. 
    -> we could have thread 0 load element [0-4], thread 2 [5-8], and so on all the way to thread 256 which would load [1021-1024], but this would not be effective.
        in this arrangement, consecutive threads would not access consecutive addresses, and we are not efficiently maxiziming occupancy. 
    -> we want, ideally, for a warp to load contigous addresses to allow for memory coalescing. we hence want thread 0 to load element 0, thread 1 to load element 1, and so on. 
        we can load all elements by setting the stride to be threadsPerBlock, such that thread 0 will load element 0 and threadsPerBlock, thread 1 element 1 and threadsPerBlock + 1, 
        all the way up to some thread loading IN_TILE_DIM * IN_TILE_DIM.
    */
   const int tid = threadIdx.y * OUT_TILE_DIM + threadIdx.x;

   for (int i{tid}; i < IN_TILE_DIM * IN_TILE_DIM; i+=threadsPerBlock)
   {
    /* inside this loop, we want to shift startCol and startRow to load all elements we need. */
    // getting the row index in shared memory:
    int ty = i / IN_TILE_DIM;
    // getting the col index in shared memory
    int tx = i % IN_TILE_DIM;
    int sharedRow = startRow + ty;
    int sharedCol = startCol + tx;

    if (sharedRow >= 0 && sharedRow < height && sharedCol >= 0 && sharedCol < width)
    {
        N_s[ty][tx] = N[sharedRow * width + sharedCol];
    }
    else
    {
        N_s[ty][tx] = 0.0f;
    }
   }
   __syncthreads();
   // output computation:
   const int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
   const int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y;

   if (row < height && col < width)
   {
    float Pval = 0.0f;
    #pragma unroll
    for (int fRow{0}; fRow < 2*FILTER_RADIUS+1; ++fRow)
    {
        #pragma unroll
        for (int fCol{0}; fCol < 2 * FILTER_RADIUS+1; ++fCol)
        {
            Pval += F_c[fRow][fCol]*N_s[threadIdx.y + fRow][threadIdx.x + fCol];
        }
    }
    P[row * width + col] = Pval;
   }
}
