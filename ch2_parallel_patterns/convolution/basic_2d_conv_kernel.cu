#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>

/*
input size 64.0 MiB

global, runtime radius      1.9914 ms     421.2 GFLOP/s   speedup 1.00x   max err 9.75e+00
constant memory             1.3378 ms     627.0 GFLOP/s   speedup 1.49x   max err 9.75e+00
*/


#define FILTER_RADIUS 2
#define FILTER_WIDTH (2 * FILTER_RADIUS + 1)

// first implementation: basic one, no use of constant memory or any other optimization. filter in global memory and runtime radius. 
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
            int inRow = row - r + filter_y;
            int inCol = col - r + filter_x; 
            
            if (inRow < height && inRow >= 0 && inCol < width && inCol >= 0)
            {
                Pval += F[filter_y * filter_width + filter_x]*N[inRow * width + inCol];
            }
        }
    } 
    P[row * width + col] = Pval;
}

// using constant memory for the filter
// the convolution filter is a good candidate for a variable stored in constant memory:
/*
-> the size of F is generally relatively small (m << n, the filter being mxm and the input/output arrays being nxn). 
-> the contents of the filter do not change through the execution of the filter. 
-> all threads access all the filter elements, in the same order. 
*/
__constant__ float F_const[FILTER_WIDTH * FILTER_WIDTH];

__global__ void convolutional_2D_constant(float* N, float* P, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width) return;

    float Pval = 0.0f;

    for (int filter_y{0}; filter_y < FILTER_WIDTH; ++filter_y)
    {
        #pragma unroll
        for (int filter_x{0}; filter_x < FILTER_WIDTH; ++filter_x)
        {
            int inRow = row - FILTER_RADIUS + filter_y;
            int inCol = col - FILTER_RADIUS + filter_x; 
            
            if (inRow < height && inRow >= 0 && inCol < width && inCol >= 0)
            {
                Pval += F_const[filter_y * FILTER_WIDTH + filter_x]*N[inRow * width + inCol];
            }
        }
    } 
    P[row * width + col] = Pval;
}

// just for reference: CPU
static void conv2D_cpu(const float* N, const float* F, float* P,
                       int r, int width, int height)
{
    int fw = 2 * r + 1;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            float acc = 0.0f;
            for (int fy = 0; fy < fw; ++fy) {
                for (int fx = 0; fx < fw; ++fx) {
                    int inRow = row - r + fy;
                    int inCol = col - r + fx;
                    if (inRow >= 0 && inRow < height &&
                        inCol >= 0 && inCol < width) {
                        acc += F[fy * fw + fx] * N[inRow * width + inCol];
                    }
                }
            }
            P[row * width + col] = acc;
        }
    }
}

static double max_abs_diff(const float* a, const float* b, size_t n)
{
    double worst = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

int main(int argc, char** argv)
{
    const int width = (argc > 1) ? atoi(argv[1]) : 4096;
    const int height = width;
    const int r = FILTER_RADIUS;
    const int fw = FILTER_WIDTH;

    const int WARMUP = 10;
    const int ITERS = 100;

    const size_t n_elems = (size_t)width * height;
    const size_t n_bytes = n_elems * sizeof(float);
    const size_t f_bytes = (size_t)fw * fw * sizeof(float);

    printf("input %d x %d, filter %d x %d (radius %d)\n", width, height, fw, fw, r);
    printf("input size %.1f MiB\n\n", n_bytes / (1024.0 * 1024.0));

    float* h_N = (float*)malloc(n_bytes);
    float* h_F = (float*)malloc(f_bytes);
    float* h_P = (float*)malloc(n_bytes);
    float* h_ref = (float*)malloc(n_bytes);
    if (!h_N || !h_F || !h_P || !h_ref) { fprintf(stderr, "host alloc failed\n"); return 1; }

    srand(1234);
    for (size_t i{0}; i < n_elems; ++i)
    {
        h_N[i] = (float)rand() / (float)RAND_MAX;
    }
    for (int i{0}; i < fw * fw; ++i)
    {
        h_F[i] = (float)rand() / (float)RAND_MAX;
    }
    conv2D_cpu(h_N, h_F, h_ref, r, width, height);

    // allocate to device
    float* d_N = nullptr;
    float* d_F = nullptr;
    float* d_P = nullptr;

    cudaMalloc(&d_N, n_bytes);
    cudaMalloc(&d_F, f_bytes);
    cudaMalloc(&d_P, n_bytes);

    cudaMemcpy(d_N, h_N, n_bytes, cudaMemcpyHostToDevice);
    // just for the non-constant memory version
    cudaMemcpy(d_F, h_F, f_bytes, cudaMemcpyHostToDevice);

    // the constant memory step:
    cudaMemcpyToSymbol(F_const, d_F, f_bytes);

    dim3 block(16, 16);
    dim3 grid((width - block.x + 1) / block.x, (height - block.y + 1) / block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const double gflop = 2.0 * (double)n_elems * fw * fw / 1e9;

    float ms_base = 0.0f;
    const char* names[2] = { "global, runtime radius  ",
                             "constant memory         " };

    convolutional_2D_base_kernel<<<grid, block>>>(d_N, d_F, d_P, r, width, height);
    convolutional_2D_constant<<<grid, block>>>(d_N, d_P, width, height);

    for (int k = 0; k < 2; ++k) {
        cudaMemset(d_P, 0, n_bytes);

        for (int i = 0; i < WARMUP; ++i) {
            if (k == 0) convolutional_2D_base_kernel<<<grid, block>>>(d_N, d_F, d_P, r, width, height);
            else        convolutional_2D_constant<<<grid, block>>>(d_N, d_P, width, height);
           
        }
        cudaGetLastError();
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < ITERS; ++i) {
            if      (k == 0) convolutional_2D_base_kernel<<<grid, block>>>(d_N, d_F, d_P, r, width, height);
            else           convolutional_2D_constant<<<grid, block>>>(d_N, d_P, width, height);
        } 
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float total_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, start, stop);
        float ms = total_ms / ITERS;
        if (k == 0) ms_base = ms;

        cudaMemcpy(h_P, d_P, n_bytes, cudaMemcpyDeviceToHost);
        double err = max_abs_diff(h_P, h_ref, n_elems);

        printf("%s  %8.4f ms   %7.1f GFLOP/s   speedup %.2fx   max err %.2e\n",
               names[k], ms, gflop / (ms / 1e3), ms_base / ms, err);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_N);
    cudaFree(d_F);
    cudaFree(d_P);
    free(h_N); free(h_F); free(h_P); free(h_ref);

    return 0;
}
    



