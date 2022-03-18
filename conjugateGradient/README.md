# Explanation:
 Computation has the following format:
  - All necessary data is either shared_compute (sm_ptr) and reg_compute(reg_ptr). The result is stored to result register (register)

 PERKS (currently) based on assumptions:
  1. Computation do not need to use all the shared memroy and register to achieve peak performance
  2. The bottleneck is global memory bandwith (L2 cache should also work). (If the bottleneck moved to shared memroy or register, need additional trick to balance the register and shared memory usage.) 
  3. Use remainling on-chip memory space to "cache" or "hold" data.

# RUN parameter:
  --mtx=[matrix file name]
  
  --iters=[max iteration]
  
  --fp32 (single precision)
  
  --check (check result)
  
  --vdata (use virtual data)
  
  --baseline (nothing cache version)
  
  --cmat (cache matrix)
  
  --cvec (cache vector)
  
  (default cache tb level search result of spmv)
