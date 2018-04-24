// assert ref : https://stackoverflow.com/questions/3767869/adding-message-to-assert?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
// cpu prefix sum ref: https://www.geeksforgeeks.org/prefix-sum-array-implementation-applications-competitive-programming/ 
// performance meaurement ref: https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf
// print from cuda kernel function - http://15418.courses.cs.cmu.edu/spring2013/article/15
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <math.h>
#include "helper/wtime.h"
using namespace std;

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

void fillPrefixSum(int arr[], int n, int prefixSum[])
{
    prefixSum[0] = arr[0];
    for (int i = 1; i < n; i++)
        prefixSum[i] = prefixSum[i-1] + arr[i];
}

__device__ int res=0;           //result from one block to next block
__device__ int inc=0;
__shared__ int smem[128];  

__global__ void prefix_scan_kernel (int *b_d, int *a_d, int n, int depth) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; 
    int d = 0;
    int offset = 0;

    while (tid < n) {
        smem[threadIdx.x] = a_d[tid];       // each thread copy data to shared memory
        __syncthreads();                    // wait for all threads

        //if (tid%16384 == 0 ) {   smem[tid] += res; __syncthreads();  } // result are written at the end*  

        offset = 1;                 //1->2->4->8
        for (d =0; d < depth ; d++) {                    

            if (threadIdx.x >= offset) {  
                smem[threadIdx.x] += smem[threadIdx.x-offset] ;           //after writing to smem do synchronize
                __syncthreads();      
            } // end if

            offset *=2;
        } // end for loop

        b_d[tid] = smem[threadIdx.x];        // *write the result to array b_d[tid] location
        __syncthreads();                    // wait for all threads to write results
        
        //if ((tid + 1) % 16384 == 0) { inc++; printf("\n incremented %d times\n", inc);}
        tid += 16384;               //there are no actual grid present, we just increment the tid to fetch next elemennts from input array.
        
    } // end while (tid < n)
} // end kernel function


int
main (int args, char **argv)
{
  int threadsInBlock = 128;
  int numberOfBlocks = 128;
  //int n = threadsInBlock*numberOfBlocks;
  int n = 32000000;
  int depth = log2(128);  

  int *a_cpu= (int *)malloc(sizeof(int)*n);
  int *b_cpu= (int *)malloc(sizeof(int)*n);
  int *b_ref= (int *)malloc(sizeof(int)*n);
    
  cout << "\n array is: "; 
  for (int i = 0; i < n; i++) { 
      a_cpu[i] = rand () % 5 + 2; 
      //cout << a_cpu[i] << " ";
  }   cout << endl;
  
  auto time_beg = wtime();
  fillPrefixSum(a_cpu, n, b_ref);
  auto el_cpu = wtime() - time_beg;
  
  int *a_d, *b_d; //device storage pointers

  cudaMalloc ((void **) &a_d, sizeof (int) * n);
  cudaMalloc ((void **) &b_d, sizeof (int) * n);

  cudaMemcpy (a_d, a_cpu, sizeof (int) * n, cudaMemcpyHostToDevice);
    
  auto time_beg1 = wtime();
  prefix_scan_kernel <<< numberOfBlocks,threadsInBlock >>> (b_d,a_d, n, depth);
  cudaMemcpy (b_cpu, b_d, sizeof (int) * n, cudaMemcpyDeviceToHost);
  auto el_gpu = wtime() - time_beg1;

  // cpu combines the results of each block with next block. cpu basically adds last element from previos block to
  // next element in next block. This is sequential process.
  int res = 0;
  for (int i = 0; i < n; i++) {    
         b_cpu[i]+=res;
        if((i+1)%threadsInBlock==0){ res = b_cpu[i]; }        
  }
  cout << "\n CPU Result is: "; 
  for (int i = 0; i < n; i++) {    
      //cout << b_ref[i] << " ";   
  }  cout << endl;
    
  cout << "\n GPU Result is: ";
  for (int i = 0; i < n; i++) {    
      //assert(b_ref[i] == b_cpu[i]);
      //ASSERT(b_ref[i] == b_cpu[i], "Error at i= " << i);  
      //cout << b_cpu[i] << " ";  
  } cout << endl;
    
  cout << "CPU time is: " << el_cpu * 1000 << " mSec " << endl;
  cout << "GPU kernel time is: " << el_gpu * 1000 << " mSec " << endl; 
  return 0; 
}
