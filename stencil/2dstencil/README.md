# Explanation:
 Computation is based on one assumption:
  1. All necessary data is either shared_compute (sm_ptr) and reg_compute(reg_ptr). The result is generated in result register (register)
 
 PERKS (currently) based on assumptions:
  1. Computation do not need to use all the shared memroy and register to achieve peak performance
  2. The bottleneck is global memory bandwith (L2 cache should also work). (If the bottleneck moved to shared memroy or register, need additional trick to balance the register and shared memory usage.) 
  3. Use remainling on-chip memory space to "cache" or "hold" data.
    
# Setting
2d stencil 
Can config the desired setting in [stencil type]/config.cuh 
  - Halo: the halo reagion for the [stencil type]
  - BOX: define it if the shape is BOX
  - RTILE_Y: compute unit in y axle
  - TILE_X:  compute unit in x axle
  
 Configuration for PERKS is in [stencil type]/genconfig.cuh
  - Need this setting to determine the total register usage. 
  - Currently, only total_reg=reg_fold*RTILE_Y is supported. 

 # Usage:
  run ./config.sh 
  
  run ./build.sh
  
  the builded excutable files is generated in ./build/init/
  
 # TODO
  1. automatically set register amount (feasically doable), can be refer to ./ProcessCompilerLog.py [WORKING ON IT]
  2. better print method
  3. portability (**CAUTION**) REF to BUG
 
 # BUG 
- [BUG for portability] only width_x=grid.x*[interger] is supported. 
- [BUG for portability] For PERKS with cache version only width_y=grid.y*RTILE_Y*[integer] is supported 

