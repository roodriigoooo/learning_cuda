#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

const int TILE_WIDTH{16};

/*
========================================================
  CUDA Corner-Turning Matrix Multiplication Benchmark
  Matrix Dimensions: J=2048, K=2048, L=2048
========================================================

Results Verification: PASSED (Outputs Match)

| Kernel Version       | Avg Time (ms) | Performance (GFLOPS) |
|----------------------|---------------|----------------------|
| Uncoalesced Baseline |        29.959 |              573.439 |
| Corner-Tuned         |        27.164 |              632.457 |

Speedup achieved with Corner Turning: 1.103x
*/

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

       unsigned int N_column = bx * TILE_WIDTH + ty;
       if (N_column < l && (TILE_WIDTH *ph + tx) < k) 
       /*
       note: if N is a column-major matrix with k rows, then the linear index of row r and column c is given by N[(k*c)+r]
       and hence, if we want to coalesce accesses to consecutive elements of N, we want to have consecutive threads (consecutive tx) map to that '+r'. 
       therefore, we wish to access N with N[(k*Col + TILE_WIDTH+ph) + tx]

       note: we want to assign this value not to N[ty][tx], but to N[tx][ty] so as to transpose the view of the stored shared matrix and keep things consistent. 
       
       */
       {
           // Nds[ty][tx] = N[(TILE_WIDTH * ph + ty)*l + Col];
           Nds[tx][ty] = N[k*N_column + TILE_WIDTH*ph + tx];
       }
       else Nds[tx][ty] = 0.0f;
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

// IMPLEMENTATION FINISHES HERE. ALL THE REST IS JUST BENCHMARKING AND COMPARISONS...

__global__ void TiledMatrixMulUncoalescedKernel(float* M, float* N, float* P, int j, int k, int l)
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
            Nds[ty][tx] = N[k*Col + TILE_WIDTH * ph + ty];
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

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                      << " at line " << __LINE__ << std::endl;                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


    // just for verification
void matrixMulCPU(const std::vector<float>& M, const std::vector<float>& N, std::vector<float>& P, int j, int k, int l)
{
    for (int r = 0; r < j; ++r) {
        for (int c = 0; c < l; ++c) {
            float sum = 0.0f;
            for (int i = 0; i < k; ++i) {
                // M is Row-Major: M[r * k + i]
                // N is Column-Major: N[c * k + i]
                sum += M[r * k + i] * N[c * k + i];
            }
            P[r * l + c] = sum;
        }
    }
}

int main()
{
    // Matrix dimensions: M (j x k), N (k x l) -> P (j x l)
    const int J = 2048;
    const int K = 2048;
    const int L = 2048;

    std::cout << "========================================================\n";
    std::cout << "  CUDA Corner-Turning Matrix Multiplication Benchmark\n";
    std::cout << "  Matrix Dimensions: J=" << J << ", K=" << K << ", L=" << L << "\n";
    std::cout << "========================================================\n\n";

    size_t bytes_M = J * K * sizeof(float);
    size_t bytes_N = K * L * sizeof(float);
    size_t bytes_P = J * L * sizeof(float);

    // Host allocations
    std::vector<float> h_M(J * K);
    std::vector<float> h_N(K * L);
    std::vector<float> h_P_uncoalesced(J * L, 0.0f);
    std::vector<float> h_P_corner(J * L, 0.0f);

    // Initialize host matrices
    for (int i = 0; i < J * K; ++i) h_M[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * L; ++i) h_N[i] = static_cast<float>(rand()) / RAND_MAX;

    // Device allocations
    float *d_M, *d_N, *d_P;
    CUDA_CHECK(cudaMalloc(&d_M, bytes_M));
    CUDA_CHECK(cudaMalloc(&d_N, bytes_N));
    CUDA_CHECK(cudaMalloc(&d_P, bytes_P));

    CUDA_CHECK(cudaMemcpy(d_M, h_M.data(), bytes_M, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, h_N.data(), bytes_N, cudaMemcpyHostToDevice));

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((L + TILE_WIDTH - 1) / TILE_WIDTH, (J + TILE_WIDTH - 1) / TILE_WIDTH);

    // Timing setup
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 50;

    // Benchmark 1: Uncoalesced Kernel
    // Warmup
    TiledMatrixMulUncoalescedKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, J, K, L);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        TiledMatrixMulUncoalescedKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, J, K, L);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_uncoalesced_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time_uncoalesced_ms, start, stop));
    time_uncoalesced_ms /= iterations;

    CUDA_CHECK(cudaMemcpy(h_P_uncoalesced.data(), d_P, bytes_P, cudaMemcpyDeviceToHost));

    // Benchmark 2: Corner-Tuned Kernel
    // Warmup
    TiledMatrixMulWithCornerTurningKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, J, K, L);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        TiledMatrixMulWithCornerTurningKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, J, K, L);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float time_corner_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&time_corner_ms, start, stop));
    time_corner_ms /= iterations;

    CUDA_CHECK(cudaMemcpy(h_P_corner.data(), d_P, bytes_P, cudaMemcpyDeviceToHost));

    // -------------------------------------------------------------------------
    // Correctness Verification
    bool match = true;
    for (int i = 0; i < J * L; ++i) {
        if (std::abs(h_P_uncoalesced[i] - h_P_corner[i]) > 1e-3f) {
            match = false;
            break;
        }
    }


    // Performance Reporting
    double total_flops = 2.0 * static_cast<double>(J) * K * L;
    double gflops_uncoalesced = (total_flops / (time_uncoalesced_ms / 1000.0)) / 1e9;
    double gflops_corner = (total_flops / (time_corner_ms / 1000.0)) / 1e9;
    double speedup = time_uncoalesced_ms / time_corner_ms;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Results Verification: " << (match ? "PASSED (Outputs Match)" : "FAILED (Mismatch Detected)") << "\n\n";
    std::cout << "| Kernel Version       | Avg Time (ms) | Performance (GFLOPS) |\n";
    std::cout << "|----------------------|---------------|----------------------|\n";
    std::cout << "| Uncoalesced Baseline | " << std::setw(13) << time_uncoalesced_ms << " | " << std::setw(20) << gflops_uncoalesced << " |\n";
    std::cout << "| Corner-Tuned         | " << std::setw(13) << time_corner_ms << " | " << std::setw(20) << gflops_corner << " |\n\n";
    std::cout << "Speedup achieved with Corner Turning: " << speedup << "x\n";


    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_P));

    return 0;
}
