// This program is limited to 16 array indexes only.
// BlockDimx= 8 and the GridDimX=2. This is a simple program to start with.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <chrono>
#include "helper/wtime.h"
using namespace std;
  __device__ int res;  //result from one block to next block


__global__ void vec_mult_kernel (int *b_d, int *a_d, int n) {
int tid = blockIdx.x* blockDim.x+ threadIdx.x; // initialize with block number. Tid = 0 -> 10240
__shared__ int smem[256];
  int depth = 3;    //log(blockDim.x) = log(8) = 3
  int d =0;
  int offset = 0;
  
  while (tid < n) {
  smem[tid] = a_d[tid];   // copy datat to shared memory
  __syncthreads(); //wait for all threads

  if (tid%blockDim.x == 0 ) { smem[tid] = a_d[tid]; b_d[tid] = smem[tid]+res;}
  offset = 1; //1->2->4
  for (d =0; d < depth ; d++){                        // depth = 3
    
    if (tid%blockDim.x >= offset){
  
     
      smem[tid] += smem[tid-offset] ;           //after writing to smem do synchronize
      __syncthreads();      
       
    }// end if
    offset *=2;
   } // end for 
   b_d[tid] = smem[tid] + res; // add this part  // save result to b_d after adding res to it;
  if(tid%blockDim.x == blockDim.x-1) {res = b_d[tid];}  // if last thread in block save cout
  __syncthreads();
  tid += blockDim.x;
} // end while (tid < n)
} // end kernel function



int
main (int args, char **argv)
{
// configure matrix dimensions
int n = 64;
int *a= (int *)malloc(sizeof(int)*n);
int *b= (int *)malloc(sizeof(int)*n);
// Initialize matrix A and B
  cout << "array is: ";
for (int i = 0; i < n; i++) { a[i] = rand () % 5 + 2; cout << a[i] << " ";}
  cout << endl;
int *a_d, *b_d; //device storage pointers

cudaMalloc ((void **) &a_d, sizeof (int) * n);
cudaMalloc ((void **) &b_d, sizeof (int) * n);

cudaMemcpy (a_d, a, sizeof (int) * n, cudaMemcpyHostToDevice);

// perform multiplication on GPU
auto time_beg = wtime();
vec_mult_kernel <<< 8,8 >>> (b_d,a_d, n );
cudaMemcpy (b, b_d, sizeof (int) * n, cudaMemcpyDeviceToHost);
  cout << "result is: ";
for (int i = 0; i < n; i++) {  cout << b[i] << " ";}
  cout << endl;
auto el = wtime() - time_beg;
cout << "Time is: " << el << " Sec " << endl;
return 0;
}
