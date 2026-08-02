#include <cuda_runtime.h>

__global__ void convolutional_2D_base_kernel(float* N, float* F, float* P, int r, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width) return;

    float Pval = 0.0f;
    int filter_width = 2*r+1;

    for (int filter_y{0}; filter_y < filter_width; ++filter_y)
    {
        for (int filter_x{0}; filter_x < filter_width; ++filter_x)
        {
            int filter_row = row - r + filter_y;
            int filter_col = col - r + filter_x; 
            
            if (filter_row < height && filter_row >= 0 && filter_col < width && filter_col >= 0)
            {
                Pval += F[filter_y * filter_width + filter_x]*N[filter_row * width + filter_col];
            }
        }
    } 
    P[row * width + col] = Pval;
}
