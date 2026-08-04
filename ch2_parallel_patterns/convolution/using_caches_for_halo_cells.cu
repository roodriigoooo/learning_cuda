#include <cuda_runtime.h>

#define FILTER_RADIUS 2
#define OUT_TILE_DIM 32

__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void conv_using_caches_for_halo_cells(float* N, float* P, int width, int height)
{
    __shared__ float N_s[OUT_TILE_DIM][OUT_TILE_DIM];
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y;

    /*
    each thread loads exactly one element. shared memory is not OUT_TILE_DIM^2 elements, and the input tile and output tile now cover the same region of the 2d structure. 
    */
    if (row < height && col < width)
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    __syncthreads();

    if (row < height && col < width)
    {
        float Pval = 0.0f;
        for (int fRow{0}; fRow < 2 * FILTER_RADIUS + 1; ++fRow)
        {
            for (int fCol{0}; fCol < 2 * FILTER_RADIUS + 1; ++fCol)
            {
                /*
                for any given output value, some number of that thread's [FILTER_RADIUS * 2 +1]^2 footprint will come from shared memory
                (as they are part of the internal block) and some others will be read from global memory. 
                --> the latter case is the worst case. Note that a thread in the very middle of the tile will have all of its footprint inside the shared memory, and 
                    will never take a global path whatsoever. 
                --> in any case, a block's needed input portion that is not part of its shared memory is part of the territory of neighboring blocks. 
                    note: within a single block, the same halo element is needed by up to 2R+1 different threads, and those repeated reads will be caught by L1, which are private to an SM. 
                    across blocks, the sharing needs to happen at the L2 level. so say block(2,1) on SM A will load its interior, fill the L2 cache, and then Block(1,1) on SM B will eventually hit the L2 cache, 
                    the hardware will notice that the desired memory location is occupied by an element in the L2 cache, and serve SM B. DRAM is not reached in this process!!
                */
                int tileRow = threadIdx.x - FILTER_RADIUS + fRow;
                int tileCol = threadIdx.y - FILTER_RADIUS + fCol;

                // if this is true, this is not a ghost cell. 
                if (tileRow >= 0 && tileRow < OUT_TILE_DIM && tileCol >= 0 && tileCol < OUT_TILE_DIM)
                {
                    Pval += F_c[fRow][fCol]*N_s[tileRow][tileCol];
                }
                // handle ghost cells
                else
                {
                    int ghostRow = row - FILTER_RADIUS + fRow;
                    int ghostCol = col - FILTER_RADIUS + fCol;
                    if (ghostRow >= 0 && ghostRow < height && ghostCol >= 0 && ghostCol < width)
                    /*
                    THIS IS A BET RATHER THAN A GUARANTEE. nothing in CUDA really guarantees us that some blocks will be resident at the same time, or even close in time. 
                    in practice though it is a reasonable assumption to make, as blocks are dispatched in SMs in roughly linear order, so blocks adjacent in blockIdx.x will tend to launch 
                    very close together, while vertical neighbors will be a little further apart in the dispatch order. 
                    -> so for instance, in a large image with relatively small L2 capacity, this reuse might miss completely for vertical neighbors. 
                    */
                        Pval += F_c[fRow][fCol]*N[ghostRow * width + ghostCol];
                }
            }
        }
        P[row * width + col] = Pval;
    }
}
