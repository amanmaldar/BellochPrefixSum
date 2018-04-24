#!/bin/bash
git pull origin master
nvcc downsweep.cu -std=c++11 -o downsweep
#nvcc prefix_16.cu -std=c++11 -o prefix
./downsweep
