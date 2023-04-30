# PERKS
PERKS sample implementation of iterative stencil solver and conjugate gradient solver

Explanation of how to run conjugate gradient solver is in:

conjugateGradient/README.md

Explanation of how to run stencil solver is in:

stencil/2dstencil/README.md

stencil/3dstencil/README.md

You can compile code with config.sh and then build.sh in the ./conjugateGradient, ./stencil/2dstencil and ./stencil/3dstencil folders.
The executable will be in ./build/init folders. 

## Acknowledge
stencil used some code from the following repository:
- Unit test & data generation: https://github.com/pssrawat/IEEE2017
- Shared memory optimizations: https://github.com/naoyam/benchmarks

## Cite PERKS:
Lingqi Zhang, Mohamed Wahib, Peng Chen, Jintao Meng, Xiao Wang, Endo Toshio, and Satoshi Matsuoka. 2023. PERKS: a Locality-Optimized Execution Model for Iterative Memory-bound GPU Applications. In 2023 International Conference on Supercomputing (ICS ’23), June 21–23, 2023, Orlando, FL, USA. ACM, New York, NY, USA, 13 pages, https://doi.org/10.1145/3577193.3593705
