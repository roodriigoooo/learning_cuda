## Memory Architecture and Data Locality Exercises

#### *1. Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumption?*
- No, in this case using shared memory cannot reduce global memory bandwidth consumption. The opposite is true: shared memory would worsen performance due to extra instruction and synchronization overhead. In matrix addition, the operation of each element is $C_{i,j} = A_{i, j} + B_{i, j}$. That is, each elememt from each input matrix is read once from global memory, from a single thread. No thread ever needs an element loaded by another thread. 
- Shared memory is user-managed cache, and cache only benefits on resuing data across multiple threads/operations, which does not occur here. 
Notice also that matrix addition is a memory-bound operation, with low arithmetic intensity. If a thread performs one floating-point addition (1 FLOP) and reads two 4-byte floats or ints + writes another 4-byte float or int, the compute-to-global-memory-access of a thread is 1 FLOP/12 bytes. Introducing extra steps (loading global memory, writing to shared memory, reading shared memory, syncing, adding) would just add unnecessary altency without saving any global memory transactions. 


#### *3. What type of incorrect execution behavior can happen if one forgot to use one or both `__syncthreads()` in the following kernel?*

```c++
__global__ void TiledMatrixMulKernel(float* M, float* N, float* P, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y; int ty = threadIdx.y;
    int bx = blockIdx.x; int tx = threadIdx.x;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;.

    float Pvalue = 0;
    for (int ph{0}; ph < ceil(Width/(float)TILE_WIDTH); ++ph)
    {
       if (Row < Width && (TILE_WIDTH * ph + tx) < Width) 
       {
            Mds[ty][tx] = M[Width*Row + TILE_WIDTH*ph + tx];
       }
       else Mds[ty][tx] = 0.0f;
       if (Col < Width && (TILE_WIDTH *ph + ty) < Width) 
       {
            Nds[ty][tx] = N[(TILE_WIDTH * ph + ty)*Width + Col];
       }
       else Nds[ty][tx] = 0.0f;
       __syncthreads();  // here
       for (int k{0}; k<TILE_WIDTH; ++k)
       {
            Pvalue += Mds[ty][k]*Nds[k][tx];
       }
       __syncthreads(); // here
    }
    if (Row < Width && Col << Width)
        P[Row*Width+Col] = Pvalue;
}
```
- In both cases race conditions would occur. 
    - For the first call to `__syncthreads()`, the problem is that each thread will load its own dedicate element into shared memory at `Mds[ty][tx]`, but inside the multiplication loop it will read elements loaded by other threads. So if some thread A executes faster than thread B, thread A might attempt to read shared memory values that thread B has not yet written. 
    - For the second call to `__syncthreads()`, note that `Pvalue` is a scalar automatic variable, and therefore is stored in thread registers. Threads are not sharing this variable, so it is not a risk for one thread to overwrite another's. The shared memory access being guarded here is across iterations of the `ph` loop. 
        - If thread A computes its partial dot product for phase 0, and then immediately loops around to phase 1, it would attempt to overwrite shared memory array with new matrix elements. If thread B is still computing its partial dot product in phase 0, thread B would akwardly read phase 1 data in the later stages of its supposedly phase 0 partial calculations.

#### *4. Assuming capacity were not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory? Explain.*
Registers and shared memory are both on-chip memory structures, and therefore both offer relatively low access latency and are generally best for high arithmetic intensity applications when compared to constant and global memory. However, registers are thread-local. That is, if we have N threads and each stores a variable in a register, we will have N copies of that variable across all threads, but each copy will be unable to be directly accessed by threads other than that which initialized it. So, if 32 threads need a value, all 32 of them would have to fetch it from the global memory and into their own registers. Shared memory becomes valuable because its scope is the thread block, and not each specific thread. A single thread can therefore fetch a value from global memory once, and make it available for all other threads in the block.

#### *5. 5. For our tiled matrix-matrix multiplication kernel, if we use a 32x32 tile, what is the reduction of memory bandwidth usage for input matrices M and N?*
- Without tiling, calculating a single output element $P_{i,j}$ will have a thread multiply row $i$ of matrix $M$ by column $j$ of matrix $N$. That is, we will read $2 * Width$ elements from global memory to compute one element. 
- With tiling with a 32x32 tile, a block of 1024 threads loads a 32x32 tile of $M$ and another of $N$, and each one of those elements loaded is resued 32 times across the 32 threads in that block row/column. 
    - Each element is read from global memory once, and reused across the 32 blocks in a thread, a process which happens Width/TILE_WIDTH times. The global memory reads per thread is therefore $2 * \frac{Width}{TILE WIDTH}$. 
    - Therefore, global memory traffic for $M$ and $N$ drops 32-fold. 

#### *6. Assume that a CUDA kernel is launched with 1000 thread blocks each of which has 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?*
- A local automatic variable is stored on thread-local registers. We have 1000 blocks each of which has 512 threads and, therefore, $512*1000 = $512000$ versions of the variable will be created through the lifetime of the execution of the kernel, regardless of occupancy. 

#### *7. In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?*
- Shared memory has a block scope, and we have 1000 thread blocks. Therefore, 1000 versions of the variable will be created through the lifetime of the execution of the kernel.

#### *8. Consider performing a matrix multiplication of two input matrices with dimensions N×N. How many times is each element in the input matrices requested from global memory when:*
#### *a) There is no tiling?*
- See question 5 above. With no tiling, each element of the two input matrices will be requested from global memory N times, if the input matrices have dimensions NxN. 
#### *b) Tiles of size TxT are used?*
- If tiles of size TxT are used, each element in the NxN matrices will be requested from global memory N/T times. We know that TxT tiling reduces global memory requests by a factor of T, from question 5. 

#### *9. A kernel performs 36 floating point operations and 7 32-bit global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute- or memory-bound*
#### *a) Peak FLOPS = 200 GFLOPS, Peak Memory Bandwidth = 100 GB/s*
- For this device, the machine balance is 200GFLOPS/100GB/s = 2 FLOPS/byte. At the same time, the compute-to-global-memory-access of the kernel would be 36FLOPS/28 bytes = 1.286 FLOPS/byte. Therefore, since this arithmetic intensity is lower than the machine balance, this kernel would be memory-bound under these device properties. 

#### *a) Peak FLOPS = 300 GFLOPS, Peak Memory Bandwidth = 250 GB/s*
- For this device, the machine balance is 1.2 FLOPs/byte. Therefore, in this case the arithmetic intensity of the kernel exceeds the machine balance of the underlying device, and therefore is compute-bound. 


#### *10. To manipulate tiles, a new CUDA programmer has written the following device kernel which will transpose each tile in a matrix. The tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of matrix A is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. BLOCK_WIDTH is known at compile-time and could be set anywhere from 1 to 20.*
```c++
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDim(A_width/blockDim.x, A_height/blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

__global__ void BlockTranspose(float* A_elements, int A_width, int A_height)
{
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdX.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```
#### *a) Out of the possible range of values for BLOCK_SIZE, for what values of BLOCK_SIZE will this kernel function execute correctly on the device?*
- The key thing to note here is that, in line `blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];`, $thread_{x, y}$ loads an element from global memory into `blockA[ty][tx]`. The same thread, in the line immediately after, reads an element from shared memory written by $thread_{y,x}$. Therefore, the risk is that a given thread attemps to read an element from shared memory that has not yet been written by another thread. The only case in which this risk does not exist is if we let `BLOCK_SIZE` be 1, in which case a block would contain a single thread, which would not have to risk reading anything from anyone else. This is of course not very useful. 

#### *b) If the code does not execute correctly for all BLOCK_SIZE values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work with all of the possible BLOCK_SIZE values.*
- The root cause is that described in part a). As a fix to the code, a sufficient solution would be to explicitly synchronize the threads between the last two lines:
```c++
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDim(A_width/blockDim.x, A_height/blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

__global__ void BlockTranspose(float* A_elements, int A_width, int A_height)
{
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_SIZE + threadIdX.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];
    __syncthreads();
    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```

#### *11. Consider the following CUDA kernel and the corresponding host function that calls it:*
```c++
__global__ void foo_kernel(float* a, float* b)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float x[4];
    __shared__ float y_s;
    __shared__ float b_s[128];
    for (unsigned int j = 0; j < 4; ++j)
    {
        x[j] = a[j*blockDim.x*gridDim.x + i];
    }
    if (threadIdx.x == 0)
    {
        y_s = 7.4f;
    }
    b_s[threadIdx.x] = b[i];
    __syncthreads();
    b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3] + y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128];
}

void foo(int* a_d, int* b_d)
{
    unsigned int \texttts{N} = 1024;
    foo_kernel<<<(N+128-1)/128, 128>>>(a_d, b_d);
}
```

#### *a) How many versions of the variable `i` are there?*
- Being an automatic scalar variable, the `i` variable will be stored in the threads' registers. This means that there will be as many versions as there are threads. In this case, there will be 8 blocks, each with 128 threads. Therefore, there will be 1024 versions of `i`. 

#### *b) How many versions of the array `x[]` are there?*
- `x[]` is an automatic array variable, and hence it will be stored by default in the local memory, which, like registers, have a thread scope. There will be as many versions of `x[]` are there are threads (1152 versions). This is true even if the CUDA compiler decided to instead store the variable in registers in the case that all accesses to `x[]` were with constant indexes.

#### *c) How many versions of the variable `y_s` are there?`*
- `y_s` is declared with `__shared__`, and therefore it has a block scope. There will be one instance of `y_s` in each block. We have 8 blocks, so there will be 8 versions of `y_s`. 

#### *d) How many versions of the array `b_s[]` are there?*
- same as c)

#### *e) What is the amount of shared memory used per block (in B)?*
- `float y_s` is one variable of 4 bytes, and `float b_s[128]` is one array of 128 floats of 4 bytes each. Therefore, the amount of shared memory used per block will be 129*4 = 512 bytes. 

#### *f) What is the floating point to global memory access ratio of the kernel (in OP/B)?*
- In total, we access 24 bytes of global memory: in the for loop, we access some element of `a` four times (4 * 4 bytes). Then we read again `b[i]` (4 more bytes). Then a write operation happens near then end, when writing to `b[i]` (1 * 4 bytes). 
- In regards to floating-point operations, we perform 5 multiplication operations and 5 addition operations. 
- Therefore, the floating point to global memory access ratio of the kernel is 10 FLOPS/24 bytes = 5 FLOPS/12 bytes = 0.417 OP/B. 

#### *12) Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64 K (65536) registers/SM, and 96 KB of shared memory/SM. For each of the following kernel characteristics, specify if the kernel can achieve full occupancy. If not, specify the limiting factor.*
#### *a) The kernel uses 64 threads/block, 27 registers/thread, and 4 KB of shared memory/SM.*
- 64 threads/block and 32 blocks/SM -> 2048 threads per SM. good. 
- 2048 threads/SM, 27 registers/thread -> 55296 registers/SM. good. 
- 32 blocks/SM and 4 KB of shared memory/block -> 128 KB > 96 KB. The shared memory is exhausted first, the SM can only fit `floor(96KB/4KB per block) = 24 blocks`. At 24 blocks, we would have 1536 threads (occupancy of 1536/2048) = 75%.  
#### *b) The kernel uses 256 threads/block, 31 registers/thread, and 8 KB of shared memory/SM.*
- 256 threads/block and 32 blocks/SM = 8192 threads per SM. 8192 > 2048. We can have a maximum of 2048/256 = 8 blocks, but all threads are still used. good. 
- 2048 threads/SM and 31 registers/thread -> 63488 registers/SM. good. 
- 8 blocks/SM, and 8KB of shared memory/block. 64 KB of shared memory/SM. good. 
- Full occupancy is reached. 
