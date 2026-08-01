#### *1. Consider the following CUDA kernel. For each of the folowing memory accesses, specify whether they are coalesced, uncoalesced, or if coalescing is not applicable:*
```c++
__global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e)  // 01
{
    unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;                       // 02
    __shared__ float a_s[256];                                                // 03
    __shared__ float bc_s[4*256];                                             // 04
    a_s[threadIdx.x] = a[i];                                                  // 05
    for (unsigned int j = 0; j < 4; ++j)                                      // 06
    {
        bc_s[j*256 + threadIdx.x] = b[j*blockDim.x*gridDim.x + i] + c[i*4 + j];  // 07
    }
    __syncthreads();                                                          // 08
    d[i+8] = a_s[threadIdx.x];                                                // 09
    e[i*8] = bc_s[threadIdx.x*4];                                             // 10
}
```
- #### *a. The access to array `a` of line 05*
    - Yes, accesses to array `a` in line 05 can be coalesced. We see that consecutive threads (consecutive values of `i`) access consecutive values in `a`. 

- #### *b. The access to array `a_s` of line 05*
    - No. Coalescing is not applicable here. Global memory coalescing rules do not apply in shared memory, which is a faster access memory structure and hence does not rely on DRAM bursts to make focused use of data. 

- #### *c. The access to array `b` of line 07*
    - Yes. `b` is an array stored in global memory (accesses to it are not done with constant indexes, and hence the compiler can't optimize the array as constant stored in a register) and the term `j*blockDim.x*gridDim.x` is uniform across all threads. That is, it will evaluate to the same value for all threads. We also know that the `+i` term will make the final index value consecutive for consecutive threads, so threads read contiguous locations in global memory. 

- #### *d. The access to array `c` of line 07*
    - No. Even if `c` is an array stored in global memory and `j` is a uniform value across all threads, the term `i*4` forces consecutive threads to access non-contiguous locations in memory. For instance, keeping `j` fixed at 0, thread 0 will access `b[0]`, thread 1 will access `b[4]`, thread 2 will access `b[8]`, and so on. We see that consecutive threads access memory locations 16 bytes apart. 

- #### *e. The access to array `bc_s` of line 07*
    - No. Shared memory. 

- #### *f. The access to array `a_s` of line 09*
    - No. Shared memory. 

- #### *g. The access to array `d` of line 09*
    - Yes. a constant offset of `+8` does not affect consecutive threads accessing contiguous elements in global memory `d[i+8]`, `d[i+9]`, `d[i+10]`, .... 

- #### *h. The access to array `bc_s` of line 10*
    - No. Shared memory. 

- #### *i. The access to array `e` of line 10*
    - No. See d.

#### *2. Write a matrix multiplication kernel function that uses corner turning, corresponding to the design illustrated here:* 

![Applying corner turning to coalesce accesses to matrix N which is stored in column-major layout.](figures/image.png)[^1]

[^1]: Hwu, W. W., Kirk, D. B., & El Hajj, I. *Programming Massively Parallel Processors: A Hands-on Approach* (5th ed.). Morgan Kaufmann.

- See my [solution](https://github.com/roodriigoooo/learning_cuda/blob/main/ch1_fundamental_concepts/performance_considerations/mat_mul_with_corner_turning.cu)

#### *3. For tiled matrix multiplication, out of the possible ranges of values for `BLOCK_SIZE`, for what values of `BLOCK_SIZE` will the kernel completely avoid uncoalesced accessed to global memory (You need to consider only square blocks.)*

- To answer this question, we need to consider how global memory is accessed in the kernel. When loading matrix `M` to shared memory, the index of the loaded element is given by: `(by * BLOCK_SIZE + ty)*Width + (ph * BLOCK_SIZE + tx)`, where `(by * BLOCK_SIZE + ty)` is the accessed row in `M`. For `N`, the index of the loaded element is given by:  `(ph * BLOCK_SIZE + ty)*Width + (bx * BLOCK_SIZE + tx)`. In any case, we see that for all `M`, `N` and output matrix `P`, the `tx` serves as the unit-stride term that moves contiguously across locations, while the `ty` term is multiplied by Width, and therefore moves in jumps of `Width * 4` bytes.
- Therefore, any BLOCK_SIZE that does not fully make use of the number of threads in a warp (32) is bound to have some uncoalesced accesses. If `BLOCK_SIZE` was to be 32 (or a multiple thereof), warp 0 will always have `ty = 0`, warp 1 `ty=1`, etc, and `tx` would run contigously for all threads. This would allow the hardware to combine the 32 accesses into a single 128-byte transaction. 
- Any multiple of 32 for `BLOCK_SIZE` would allow this, but we also need to keep in mind the cap on maximum threads (assume 1024 threads per block max.). `BLOCK_SIZE^2` is the number of threads we would have in a block of size `BLOCK_SIZE` (assuming squared blocks) and therefore a `BLOCK_SIZE` of 64 would already far surpass the limit (4096 threads). 

#### *4. Implement a vector addition kernel that uses vector loads and handles the boundary conditions correctly*. 

- See my [solution](https://github.com/roodriigoooo/learning_cuda/blob/main/ch1_fundamental_concepts/performance_considerations/vector_loads_add.cu)

#### *5. Consider the following shared memory array and access to the array by a block with one warp (32 threads):*
```c++
__shared__ float a[1024];
a[threadIdx.x*stride] = ...;
```
#### *Assume that `a[0]` is placed in bank 0. For each of the following values of `stride`, specify the banks accessed by the warp, and the number of bank conflicts on each bank (if any):*

    - Firstly, to answer this question we need to recall that shared memory is made from SRAM technology, which has much shorter latency and can therefore avoid relying on bursts and keep banks narrow. Assuming, as is typical, that the shared memory structure has 32 banks, each 32 bits wide, we will therefore have a single `float` element (4 bytes) at any given bank. 
    - In general, the bank used, under these assumptions, will just be the index mod 32: `bank(a[i]) = i % 32`. So, for instance, if `stride` = 12, then thread `t` will read `a[12t]` and the bank for thread 0 will be 0, that of thread 1 will be 12, that of thread 4 will be 16, etc. Thread 8 will have index 96 (8*12), and the bank will be, once again, 0. Not only that, but no new bank will ever appear after this, as thread 9 is just thread 1 shifted by 32 * 3 = 96, just like thread 12 is thread 4 shifted by 96, and 96 % 32 = 0. 
    - Put another way, if we take 12-unit long steps around a 32-unit circular strip, we will first land on 12, then on 24, and then 36 will wrap around to land us at 4. 36 is a multiple of our step size, and we will in fact always land at an index of shape "some multiple of step size - some multiple of 32". Once we have reached 4 we will also be able to reach 16, 28, 8, 20, 32, 12, 24, 4. **We will never reach a point that is not a multiple of 4**, and there are 8 multiples of 4 that are below 32 (32/4 = 8), which is the number of banks we can possibly make use of given a stride of 12. 
    - In regards to the number of bank conflicts per bank, we know that whatever the number of banks used is, they will be used equally among all threads. If we use 1 bank, then there will be 32 bank conflicts on that bank. If we use 2 banks, there will be 16 bank conflicts on each bank, since 32 / (32/16) = 16. So, in general, the number of bank conflicts per thread will be given by 32 / (32/gcd(stride, 32)) = gcd(stride, 32). 

- #### *a. `stride` = 32*
    - Banks accessed: only bank 0 (since all accesses will be exact multiples of 32). 
    - Number of bank conflicts = 32. 
- #### *b. `stride` = 31*
    - Banks accessed: all 32 banks. Thread 1 will go to bank 31, thread 2 to bank 30, and so on. Completely conflict free. 
- #### *c. `stride` = 24*
    - Banks accessed: thread 1 will go to bank 24, thread 2 will go to bank 16, thread 3 will go to bank 8. thread 4 will go to bank 0, and no new bank will appear from this point onwards. 4 banks. 
    - 8 bank conflicts per bank. 
- #### *d. `stride` = 16*
    - Banks accessed: two banks: 0 and 16. 
    - 16 bank conflicts per bank. 
- #### *e. `stride` = 12*
    - Banks accessed: 8 banks, see discussion above. 
    - 4 bank conflicts per bank. 
- #### *f. `stride` = 8*
    - Same pattern as 24, as gcd(8, 32) = gcd(24, 32) = 8.
- #### *g. `stride` = 7*
    - gcd(7, 32) = 1. All banks will be accessed, completely conflict free. 



