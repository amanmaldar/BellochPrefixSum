#!/bin/bash
git pull origin master
nvcc prefix.cu -std=c++11 -o prefix
#nvcc prefix_16.cu -std=c++11 -o prefix
./prefix
