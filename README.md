### Belloch Prefix Sum Implementation using CUDA
  
This assignement works with parallel prefix scan using upsweep and downsweep approach. The upsweep uses inclusive scan while downsweep uses exclusive scan. The CPU results are generated using exclusive scan and compared against GPU results.

#####  Approach:
- Asssume we have array A of size 64. blockDim.x = 8, gridDim.x = 8 
- Copy A to GPU DRAM as A_D. Copy A_D to correpsonding shared memory of each block.
- Each block runs the upsweep addtion. We are interested in last element in each block.
- Copy this last element from each block to CPU. 
- CPU generates the cummulative addition for this block. The results are copied back to GPU as blocksum_device.
- Load nth block's last element with blocksum_device[n-1] element. Do this for all blocks.
- Perform the downsweep on all the blocks now. Copy the result to CPU. Compare with reference CPU generated results.

##### Conclusion:
Downsweep kernel takes less time than upsweep kernel. 

##### Result 1:

##### Input array is: 
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
##### CPU Reference Result is: 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 
##### Blocksum_CPU Result is: 
8 16 24 32 40 48 56 64 
##### GPU Upsweep Result is: 
1 2 1 4 1 2 1 8 1 2 1 4 1 2 1 8 1 2 1 4 1 2 1 8 1 2 1 4 1 2 1 8 1 2 1 4 1 2 1 8 1 2 1 4 1 2 1 8 1 2 1 4 1 2 1 8 1 2 1 4 1 2 1 8 
##### GPU Downsweep (final) Result is:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 

Total entries: 64\
Total CPU version time is: 0 mSec \
GPU upsweep kernel time is: 0.0398159 mSec \
GPU downsweep kernel time is: 0.0219345 mSec \

##### ----------------------------------------------------------------------------------------------------------------------------------
##### Result 2:

Total entries: 32000000 \
Total CPU version time is: 89.1559 mSec \
GPU upsweep kernel time is: 45.217 mSec \
GPU downsweep kernel time is: 24.3959 mSec \

##### ----------------------------------------------------------------------------------------------------------------------------------

##### Result 3:

Total entries: 64000000\
Total CPU version time is: 177.801 mSec \
GPU upsweep kernel time is: 89.3421 mSec \
GPU downsweep kernel time is: 48.7161 mSec \

##### ----------------------------------------------------------------------------------------------------------------------------------
