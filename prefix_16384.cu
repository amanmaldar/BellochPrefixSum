// limited to 16284 enetries. 128 x 128 kernel is used. 16384 elements are copied to memory and each block performs klogg alogo on 128 
// elements . results are pushed back to cpu and cpu performs the final addition.
// question - i want to have more than 128 x 128 elements. lets say 128 x 128 x 4. we will see how kernel performs in next program
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include <math.h>
#include "helper/wtime.h"
using namespace std;

void fillPrefixSum(int arr[], int n, int prefixSum[])
{
    prefixSum[0] = arr[0];
    // Adding present element with previous element
    for (int i = 1; i < n; i++)
        prefixSum[i] = prefixSum[i-1] + arr[i];
}

__device__ int res=0;  //result from one block to next block
__device__ int smem[16384]; // 128*128

__global__ void vec_mult_kernel (int *b_d, int *a_d, int n, int depth) {
  
int tid = blockIdx.x* blockDim.x+ threadIdx.x; 

  int d = 0;
  int offset = 0;
    
  while (tid < n) {
      smem[tid] = a_d[tid];   // copy data to shared memory
  __syncthreads(); //wait for all threads

  if (threadIdx.x == 0 ) {   b_d[tid] = smem[tid]; }  

  offset = 1; //1->2->4
  for (d =0; d < depth ; d++){                        // depth = 3
    
    if (tid%blockDim.x >= offset){  
     
      smem[tid] += smem[tid-offset] ;           //after writing to smem do synchronize
      __syncthreads();      
       
    }// end if
    offset *=2;
   } // end for 
   b_d[tid] = smem[tid]; //+ res; no need as we alreasy are adding result to element zero above **  // save result to b_d after adding res to it;
      __syncthreads();
  
  tid += gridDim.x*blockDim.x;

} // end while (tid < n)
} // end kernel function


int
main (int args, char **argv)
{
  int threadsInBlock = 128;
  int numberOfBlocks = 128;
  int n = threadsInBlock*numberOfBlocks;
  //int n = 128*128*4;
  //int b_cpu[n];
  int depth = log2(threadsInBlock);    //log(blockDim.x) = log(8) = 3,  blockDim.x = threadsInBlock

  int *a_cpu= (int *)malloc(sizeof(int)*n);
  int *b_cpu= (int *)malloc(sizeof(int)*n);
  int *b_ref= (int *)malloc(sizeof(int)*n);
    
  cout << "\n array is: "; 
  for (int i = 0; i < n; i++) { a_cpu[i] = rand () % 5 + 2; cout << a_cpu[i] << " ";
                              }   cout << endl;
  
  auto time_beg = wtime();
  fillPrefixSum(a_cpu, n, b_ref);
  auto el_cpu = wtime() - time_beg;
  //cout << "CPU time is: " << el_cpu * 1000 << " mSec " << endl;
  
  cout << "\n CPU Result is: "; 
  for (int i = 0; i < n; i++) 
  { //cout << b_ref[i] << " ";   
  } cout << endl;
  
  int *a_d, *b_d; //device storage pointers

  cudaMalloc ((void **) &a_d, sizeof (int) * n);
  cudaMalloc ((void **) &b_d, sizeof (int) * n);

  cudaMemcpy (a_d, a_cpu, sizeof (int) * n, cudaMemcpyHostToDevice);

  time_beg = wtime();
  vec_mult_kernel <<< numberOfBlocks,threadsInBlock >>> (b_d,a_d, n, depth );
  cudaMemcpy (b_cpu, b_d, sizeof (int) * n, cudaMemcpyDeviceToHost);
   
    // cpu combines the results of each block with next block. cpu basically adds last element from previos block to
    // next element in next block. This is sequential process.
    int res = 0;
    for (int i=0;i<n;i++){
        if((i+1)%threadsInBlock==0){  b_cpu[i]+=res; res = b_cpu[i]; }
        if((i+1)%threadsInBlock!=0){
        b_cpu[i]+=res;
        }
    }
   auto el_gpu = wtime() - time_beg;



  cout << "\n GPU Result is: ";
  for (int i = 0; i < n; i++) {    
    assert(b_ref[i]== b_cpu[i]);   
    //cout << b_cpu[i] << " ";  
  } cout << endl;

  cout << "CPU time is: " << el_cpu * 1000 << " mSec " << endl;
  cout << "GPU time is: " << el_gpu * 1000 << " mSec " << endl; 
  return 0; //new
}
