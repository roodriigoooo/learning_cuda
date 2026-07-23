```c++
#include <cuda_runtime.h>

__global__ void foo_kernel(int* a, int* b)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadIdx.x < 40 || threadIdx.x >= 104)
    {
        b[i] = a[i] + 1;
    }
    if(i%2 == 0)
    {
        a[i] = b[i]*2;
    }
    for (unsigned int j{0}; j < 5 - (i%3); ++j)
    {
        b[i] += j;
    }
}


void foo(int* a_d, int* b_d)
{
    unsigned int N = 1024;
    foo_kernel<<<(N+128-1)/128, 128>>>(a_d, b_d);
}
```

a) **What is the number of warps per block?**
- -> the call to the kernel passes gridDim = (N+128-1)/128, where N = 1024. There are hence floor(1151/128) blocks, each with blockDim = (128, 1, 1)
// -> therefore, the number of warps per block is 4. (128/32)

b) **What is the number of warps in the grid?**
- -> there are 8 blocks, 128 threads per block (or 4 warps per block). There are hence 4 * 8 = 32 warps in the grid. 

c) **For the statement in line 8:**
- i) How many warps in the grid are active?
- - -> Warp 0 (threads 0-31) will be active. Warp 1 (threads 32-63) will be active even if not completely. 
- - -> Warp 2 (threads 64-95) will not be active. Warp 3 (threads 96-127) will be active. So 0, 2 and 3 will be active. 

- ii) How many warps in the grid are divergent?
- - -> Warp 1 has threads 32-39 take one possible path (satisfying the if statement) and threads 40-63 in the other path. Therefore it is divergent. 
- - -> Warp 3 has threads 96-103 not satisfying the if statement, and threads 104-127 satisfying it. Therefore warps 1 and 3 are divergent. 

- iii) What is the SIMD efficiency of warp 0 in block 0?
// -> 32 out of 32 possible threads are active: 100%

- iv) What is the SIMD efficiency of warp 1 of block 0?
// -> 8 out of 32 possible threads are active: 8/32 = 25%

- v) What is the SIMD efficiency of warp 3 in block 0?
// -> 24/32 = 75%. 

d) **for the statement on line 12:**
- i) How many warps in the grid are active?
// -> All of them. All warps have some number of threads with even ids.

- ii) How many warps in the grid are divergent?
// -> All of them. All warps have threads with alternating even/odd ids. 

- iii) What is the SIMD efficiency (in %) of warp 0 of block 0?
// -> in the range from 0 to 31, there are 16 even numbers and 16 odd numbers. So efficiency will be 16/32 = 50%

e) **for the loop on line 14:**
- i) How many iterations have no divergence?
- - -> (i%3) can only ever evaluate to either 0, 1 or 2. Therefore, the minimum number of iterations we have (which all threads will have, and hence that have no divergence) is when (i%3) == 2, which leads to j < 3. So 3 iterations have no divergence. 

- ii) How many iterations have divergence?
- - -> 2 iterations have divergence: the fourth and fifth. Some threads will have (i%3) == 1, in which we are led to j < 4, so there is an extra iteration there for some threads. same thing can be said about the threads having (i%2) == 0 (all multiples of 3, and 0), which will have 5 iterations. 

### Q2: For a vector addition, if the vector length is 2000, each thread calculates one output element, and thread block size is 512 threads, how many threads will be on the grid?
- For a thread block size of 512 and a vector length of 2000, if each thread takes care of one output element, we will need ceil(2000/512) = 4 blocks, each with 512 threads, so a total of 2048 threads. 

### Q3: For the previous question, how many warps would we expect to have divergence due to the boundary check on vector length?
- in this case, the boundary check would be something like if (i < 2000). We established already that there will be 2048 total threads, so the last 48 threads will not be utilized. Now, assuming a warp size of 32 there will be only one warp with divergence, as the very last warp (threads 2016-2047) will be completely inactive. 

### Q4: Consider a hypothetical block with 8 threads executing a section of code before reaching a barrier. The threads require the following amount of time (in ms) to execute the sections: `2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9` and spend the rest of their time waiting for the barrier. What percentage of the threads' local execution time is spent waiting for the barrier?
- the bottleneck here is 3.0 ms. 3.0ms * 8 (for each thread) gives us a total execution time including waiting of 24.0ms. From this, `sum(2.0, 2.3, 3.0, 2.8, 2.4, 1.9, 2.6, 2.9)` is time spent actually working ( = 19.9ms). So answer is 1 - 19.9/24 = 17.1% of time spent waiting. 

### Q5: A CUDA programmer says that if they launch a kernel with only 32 threads in each block, they can leave the `__syncthreads()` instruction out wherever barrier synchronization is needed. Is this a good idea?
- This is not a good practice. Among other reasons, independent thread scheduling means that even within a single warp (and 32 threads is one warp), the execution of one pass may be interleaved with the executuon of another pass. We therefore cannot and should not assume that the threads of one warp will rendez-vouz after divergent execution paths are executed. If all threads in a warp must complete a phase before any of them can continue working, then we should always use a warp-level barrier synchronization primitive such as `__syncwarp()` or the mentioned `__syncthreads()` at the block level. 

### Q6: A CUDA device's SM can take up to 1536 threads and up to 4 thread blocks. Which of the following would result in the most number of threads in the SM?
- #### a) 128 threads per block
- #### b) 256 threads per block
- #### c) 512 threads per block
- #### d) 1024 threads per block
- The answer is **c**. At 128 threads per block, capped at 4 thread blocks we have a maximum of 512 threads. At 256, 1024 threads is our maximum. At 1024 threads, having two thread block exceeds the limit of 1536, so the limit is also 1024 threads under option d. c is the only option that would reach maximum number of threads: 512 * 4 = 1536. 

### Q7: Assume a device allows up to 64 blocks per SM and 2048 threads per SM. Indicate which of the following assignments per SM is possible. In the cases where it is possible, indicate the occupancy level. 
- #### a) 8 blocks with 128 threads per block
    - Possible. Occupancy would be 50% (1024/2048)
- #### b) 16 blocks with 64 threads per block
    - Possible. Occupancy would be 50% (1024/2048)
- #### c) 32 blocks with 32 threads per block
    - Possible. Occupancy would be 50% (1024/2048)
- #### d) 64 blocks with 32 threads per block
    - Possible. Occupancy would be 100%
- #### e) 32 blocks with 64 threads per block.
    - Possible. Occupancy would be 100%. 

### Q8: Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM and 64K registers/SM. For each of the following kernel characteristics, specify if the kernel can achieve full occupancy. If not, specify the limiting factor. 
- #### a) The kernel uses 128 threads/block and 30 register/thread. 
    - To reach full occupancy, we would need 16 blocks. This amounts to 16 * 128 * 30 registers (61440) which is below the limit. Therefore, the kernel can achieve full occupancy. 
- #### b) The kernel uses 32 threads/block and 20 register/thread. 
    - To reach full occupancy, we would need 64 blocks, which is past the blocks/SM limit. Occupancy cannot be reached, being the limiting factor the blocks/SM. 
- #### c) The kernel uses 256 threads/block and 34 register/thread. 
    - To reach full occupancy, we would need 8 blocks. This amounts to 8 * 256 * 34 (=69632) registers, which is above our hardware limit. We cannot reach full occupancy under this kernel, the limiting factor being the register number per thread. 
