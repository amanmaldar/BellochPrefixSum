/*********************************************************************************************************
Institute:  University of Massachusetss Lowell
Course:     High Performance Computing EECE 7110
Student:    Aman Maldar
Instructor: Dr. Hang Liu

Program:    Parallel Prefix scan with CUDA
Approach:   This assignement works with parallel prefix scan using upsweep and downsweep approach. The upsweep 
            uses inclusive scan while downsweep uses exclusive scan. [1] The CPU results are generated using 
            exclusive scan and compared against GPU results.
            Simple appraoch:
            - Asssume we have array A of size 32. blockDim.x = 8, gridDim.x = 4 
            - Copy A to GPU DRAM as A_D. Copy A_D to correpsonding shared memory of each block.
            - Each block runs the upsweep addtion. We are interested in last element in each block.
            - Copy this last element from each block to CPU. 
            - CPU generated the cummulative addition for this block. The results are copied back to GPU as blocksum_device.
            - Load nth block's last element with blocksum_device[n-1] element. Do this for all blocks.
            - Perform the downsweep on all the blocks now. Copy the result to CPU. Compare with reference CPU generated results.
            

References:
[1] https://lowell.umassonline.net/bbcswebdav/pid-417917-dt-content-rid-5897641_1/courses/L2730-16402/scan.pdf
[2] Assertion: https://stackoverflow.com/questions/3767869/adding-message-to-assert?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
[3] CPU prefix sum: https://www.geeksforgeeks.org/prefix-sum-array-implementation-applications-competitive-programming/ 
[4] Performance meaurement: https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf
[5] Print from cuda kernel function - http://15418.courses.cs.cmu.edu/spring2013/article/15

*********************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <math.h>
#include "helper/wtime.h"
using namespace std;

#define printing 1

//********Assertion Defination [2]**************************************************************************
#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

/*********************************************************************************************************
Name:       fillPrefixSum
Input:      int arr[], int n, int prefixSum[]
Output:     prefixSum[]
Operation:  generates the result using CPU based recursive algorithm.
*********************************************************************************************************/

void fillPrefixSum(int arr[], int n, int prefixSum[])
{
    prefixSum[0] = 0;
    for (int i = 1; i < n; i++)
        prefixSum[i] = prefixSum[i-1] + arr[i];
}


/*********************************************************************************************************
Name:      prefix_upsweep_kernel
Input:     int *b_d, int *a_d, int n, int depth, int *blocksum_device
Output:    int *blocksum_device 
Operation: Performs the upsweep sum on each block. b_d is updated but not copied to CPU yet. 
Note:       Size of blocksum_device is same as number of blocks
*********************************************************************************************************/
__device__ int res=0;           //result from one block to next block
__device__ int inc=0;
__shared__ int smem[1024];  

__global__ void prefix_upsweep_kernel (int *b_d, int *a_d, int n, int depth, int *blocksum_device) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    int d = 0;
    int offset = 0;

    while (tid < n) {
        smem[threadIdx.x] = a_d[tid];       // each thread copy data to shared memory
        __syncthreads();                    // wait for all threads

        offset = 1;                         //1->2->4->8
        for (d = 1; d <= depth ; d++) {                    
            offset *= 2; 
            if (threadIdx.x % offset == offset-1 ){
                smem[threadIdx.x]+= smem[threadIdx.x- offset/2];
                 __syncthreads();     
            }

        } // end for loop

        b_d[tid] = smem[threadIdx.x];        // *write the result to array b_d[tid] location
        
        // copy last result element of block to corresponding block location in blocksum_device
        if (threadIdx.x == blockDim.x -1){
             blocksum_device[blockIdx.x] = smem[threadIdx.x];
        }
                
        __syncthreads();                    // wait for all threads to write results
        
        tid += blockDim.x * gridDim.x;      //there are no actual grid present, we just increment the tid to fetch next elemennts from input array.
        
    } // end while (tid < n)
} // end kernel function



/*********************************************************************************************************
Name:       prefix_downsweepsweep_kernel
Input:      int *b_d, int *a_d, int n, int depth, int *blocksum_device
Output:     int *b_d
Operation:  Clears the last element in first block. Copies last element in all remaining blocks (nth) with content from
            corresponding location in blocksum_device[n-1]. Last element in blocksum_device is not needed in exclusive scan.
Note:       blocksum_device is updated by cpu with cummulative sums, then updated blocksum_device is given as
            input to this kernel
*********************************************************************************************************/

__global__ void prefix_downsweepsweep_kernel (int *b_d, int *a_d, int n, int depth, int *blocksum_device) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    int d = 0;
    int offset = 0;

    while (tid < n) {
        smem[threadIdx.x] = b_d[tid];       // each thread copy data to shared memory from previous results b_d
         //b_d[tid] = smem[threadIdx.x];  
        if (threadIdx.x ==  blockDim.x -1 && blockIdx.x != 0){
            smem[threadIdx.x] = blocksum_device[blockIdx.x-1];
             //b_d[tid] = blocksum_device[blockIdx.x-1];  
        }
        if (tid == blockDim.x - 1 ){ // clear last entry in only first block - special case
             smem[threadIdx.x] = 0;
        }
        //  b_d[tid] = smem[threadIdx.x];  // uncomment in future for intermediate testing, comment entire for loop below.
        
        /*  if (tid =1){            // testing 
            printf("\n checking  blocksum_device[blockIdx.x] %d %d %d %d \n",  blocksum_device[0], blocksum_device[1],blocksum_device[2],blocksum_device[3] );
        }*/
        __syncthreads();                    // wait for all threads

        // previous result - 1 2 1 4 1 2 1 0 1 2 1 4 1 2 1 8 1 2 1 4 1 2 1 16 1 2 1 4 1 2 1 24
        //offset = 8;
        offset = blockDim.x;                 //8 -> 4 -> 2
        for (d = depth; d > 0 ; d--) {         // depth 3 -> 2  ->1  
        
            if (threadIdx.x % offset == offset-1 ){
                int tmp3 =  smem[threadIdx.x];
                int tmp1 =  smem[threadIdx.x- offset/2];
                smem[threadIdx.x- offset/2] = tmp3;
                __syncthreads();
                smem[threadIdx.x]+= tmp1;
                 __syncthreads();     
                //printf("\n printing first downsweep tid  %d  %d  %d  %d", tid, tmp1, tmp3, smem[threadIdx.x]);
            }
            offset /= 2;

        } // end for loop 

        b_d[tid] = smem[threadIdx.x];        // *write the result to array b_d[tid] location
      
        __syncthreads();                    // wait for all threads to write results
        
        tid += blockDim.x * gridDim.x;               //there are no actual grid present, we just increment the tid to fetch next elemennts from input array.
        
        
    } // end while (tid < n)
} // end kernel function


/*********************************************************************************************************
Name:      main
Input:     -
Output:     -
Operation:  Initializes CPU arrays. Initalize memory on device. Calls kernal function. Calculate CPU results.
*********************************************************************************************************/

int
main (int args, char **argv)
{
  int threadsInBlock = 8;
  int numberOfBlocks = 2;
  int n = threadsInBlock*numberOfBlocks;
  //int n = 32000000;
  int depth = log2(threadsInBlock);  

  int *a_cpu= (int *)malloc(sizeof(int)*n);
  int *b_cpu= (int *)malloc(sizeof(int)*n);
  int *b_ref= (int *)malloc(sizeof(int)*n);
  int *blocksum_cpu= (int *)malloc(sizeof(int)*numberOfBlocks);
    
  cout << "Input array is: " << endl; 
  for (int i = 0; i < n; i++) { 
      //a_cpu[i] = rand () % 5 + 2; 
      a_cpu[i] = 1;
      if (printing == 1)
            cout << a_cpu[i] << " ";
  }   cout << endl;
  
  auto time_beg_cpu = wtime();
  fillPrefixSum(a_cpu, n, b_ref);
  auto time_diff_cpu = wtime() - time_beg_cpu;
  

  int *a_d, *b_d, *blocksum_device; //device storage pointers

  cudaMalloc ((void **) &a_d, sizeof (int) * n);
  cudaMalloc ((void **) &b_d, sizeof (int) * n);
  cudaMalloc ((void **) &blocksum_device, sizeof (int) * numberOfBlocks);

  cudaMemcpy (a_d, a_cpu, sizeof (int) * n, cudaMemcpyHostToDevice);
    
  auto time_beg_kernel1 = wtime();
  prefix_upsweep_kernel <<< numberOfBlocks,threadsInBlock >>> (b_d,a_d, n, depth, blocksum_device);
  // copy b_d to CPU only to print upsweep result. Otherwise not needed. There would be a copy of b_d present 
  // on GPU DRAM inbetween calls to second kernel.
  cudaMemcpy (b_cpu, b_d, sizeof (int) * n, cudaMemcpyDeviceToHost);
  cudaMemcpy (blocksum_cpu, blocksum_device, sizeof (int) * numberOfBlocks, cudaMemcpyDeviceToHost);
  auto time_diff_kernel1 = wtime() - time_beg_kernel1;

  //  cpu basically adds last element from previos block to next element in next block. This is sequential process.
  // 10,10,10,10 becomes 10,20,30,40
            
  cout << "CPU Reference Result is: " << endl; 
  for (int i = 0; i < n; i++) {    
      if (printing == 1)
            cout << b_ref[i] << " ";   
  }  cout << endl;
    
     // update the blocksum here by cummulative addition
   int res = 0;
  cout << "Blocksum_CPU Result is: " << endl;
  for (int i = 0; i < numberOfBlocks; i++) {  
         res+= blocksum_cpu[i];
         blocksum_cpu[i] =res;  // array is updated here. Later copy to blocksum_device
         if (printing == 1)
              cout << blocksum_cpu[i] << " "; 
  } cout << endl;
            
  cout << "GPU Upsweep Result is: " << endl;
  for (int i = 0; i < n; i++) {     
      if (printing == 1)
              cout << b_cpu[i] << " ";  
  } cout << endl;
    
    
   // free a_d
   // now push the  blocksum_cpu again to kernel 2. It already has a name there as blocksum_device
    auto time_beg_kernel2 = wtime();

   cudaMemcpy (blocksum_device, blocksum_cpu, sizeof (int) * numberOfBlocks, cudaMemcpyHostToDevice);
  
   prefix_downsweepsweep_kernel <<< numberOfBlocks,threadsInBlock >>> (b_d,a_d, n, depth, blocksum_device);
      cudaMemcpy (b_cpu, b_d, sizeof (int) * n, cudaMemcpyDeviceToHost);
        auto time_diff_kernel2 = wtime() - time_beg_kernel2;

      //cout << "\n checking GPU copy of result+blocksum_device  is: ";
    cout << "GPU Downsweep (final) Result is:" << endl;
      for (int i = 0; i < n; i++) {    
          //assert(b_ref[i] == b_cpu[i]);
          ASSERT(b_ref[i] == b_cpu[i], "Error at i= " << i); 
          if (printing == 1)
                  cout << b_cpu[i] << " ";  
      } cout << endl;

  cout << "Total entries: " << n << endl; 
  cout << "Total CPU version time is: " << time_diff_cpu * 1000 << " mSec " << endl;
  cout << "GPU upsweep kernel time is: " << time_diff_kernel1 * 1000 << " mSec " << endl; 
  cout << "GPU downsweep kernel time is: " << time_diff_kernel2 * 1000 << " mSec " << endl; 
  return 0; 
}
