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



