# Explanation:
 Computation has the following format:
  - All necessary data is either shared_compute (sm_ptr) and reg_compute(reg_ptr). The result is stored to result register (register)

 PERKS (currently) based on assumptions:
  1. Computation do not need to use all the shared memroy and register to achieve peak performance
  2. The bottleneck is global memory bandwith (L2 cache should also work). (If the bottleneck moved to shared memroy or register, need additional trick to balance the register and shared memory usage.) 
  3. Use remainling on-chip memory space to "cache" or "hold" data.

 # Usage:
  run ./config.sh 
  
  run ./build.sh
  Cancel changes
  the builded excutable files is generated in ./build/init/
  
 # Auto register conifguration
run:
  python ProcessCompilerLog.py
  python ProcessRsts2cuh.py
can generate perksconfig.cuh
perksconfig.cuh configured max register for each type of stencil with 128 or 256 max register per thread and with single and double precision


 # LIMITATION 
- [for portability] only dimx=grid.x*[interger] is supported. 

