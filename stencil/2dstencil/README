#Setting
2d stencil 
Can config the desired setting in [stencil type]/config.cuh 
  Halo: the halo reagion for the [stencil type]
  BOX: define it if the shape is BOX
  RTILE_Y: compute unit in y axle
  TILE_X:  compute unit in x axle
 Configuration for PERKS is in [stencil type]/genconfig.cuh
  Need this setting to determine the total register usage. 
  Currently, only total_reg=reg_fold*RTILE_Y is supported. 

#Usage:
  run ./config.sh
  run ./build.sh
  the builded excutable files is generated in ./build/init/
  
#TODO
  1. automatically set register amount (feasically doable), can be refer to ./ProcessCompilerLog.py [WORKING ON IT]
  2. better print method
  3. portability (**CAUTION**)
    [BUG for portability] only width_x=grid.x*[interger] is supported. For PERKS with cache version only width_y=grid.y*RTILE_Y*[integer] is supported 

