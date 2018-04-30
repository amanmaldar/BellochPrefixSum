#!/bin/bash
git pull origin master
nvcc bellochPrefixSum.cu -std=c++11 -o bellochPrefixSum
#nvcc prefix_16.cu -std=c++11 -o prefix
./bellochPrefixSum
