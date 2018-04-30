#!/bin/bash
git pull origin master
nvcc bellochPrefixSum.cu -std=c++11 -o bellochPrefixSum
./bellochPrefixSum
