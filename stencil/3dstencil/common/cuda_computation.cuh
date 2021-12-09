
#define ISINITI (true)
#define NOTINITIAL (false)
#define SYNC (true)
#define NOSYNC (false)

//#undef TILE_Y
// #define USESM

// #ifdef USESM
//   #define USESMSET (true)
// #else
//   #define USESMSET (false)
// #endif


#ifndef BOX
  #if HALO==1
    #define stencilParaT \
      const REAL center=-1.67f;\
      const REAL west[1]={0.162f};\
      const REAL east[1]={0.161f};\
      const REAL north[1]={0.163f};\
      const REAL south[1]={0.164f};\
      const REAL bottom[1]={0.166f};\
      const REAL top[1]={0.165f};
    #endif
    #if HALO==2
      #define stencilParaT \
        const REAL center=-0.996f;\
        const REAL west[2]={0.083f,0.083f};\
        const REAL east[2]={0.083f,0.083f};\
        const REAL north[2]={0.083f,0.083f};\
        const REAL south[2]={0.083f,0.083f};\
        const REAL bottom[2]={0.083f,0.083f};\
        const REAL top[2]={0.083f,0.083f};
    #endif
    #define stencilParaList const REAL west[HALO],const REAL east[HALO],const REAL north[HALO],const REAL south[HALO],const REAL top[HALO], const REAL bottom[HALO], const REAL center
    #define stencilParaInput  west,east,north,south,top,bottom,center
#else
  #ifndef TYPE0
    #define stencilParaT \
      const REAL filter[3][3][3] = {\
        { {0.5/159, 0.7/159, 0.90/159},\
          {1.2/159, 1.5/159, 1.2/159},\
          {0.9/159, 0.7/159, 0.50/159}\
        },\
        { {0.51/159, 0.71/159, 0.91/159},\
          {1.21/159, 1.51/159, 1.21/159},\
          {0.91/159, 0.71/159, 0.51/159}\
        },\
        { {0.52/159, 0.72/159, 0.920/159},\
          {1.22/159, 1.52/159, 1.22/159},\
          {0.92/159, 0.72/159, 0.520/159}\
        }\
      };
  #else
    #ifdef POISSON
      #define stencilParaT \
      const REAL filter[3][3][3] = {\
        { {0,         -0.0833f,   0},\
          {-0.0833f,  -0.166f,    -0.0833f},\
          {0,         -0.0833f,   0}\
        },\
        { {-0.0833f,        -0.166f,   -0.0833f},\
          {-0.166f, 2.666f,     -0.166f},\
          {-0.0833f,       -0.166f,    -0.0833f}\
        },\
        { {0,         -0.0833f,   0},\
          {-0.0833f,  -0.166f,    -0.0833f},\
          {0,         -0.0833f,   0}\
        }\
      };
    #else
      #define stencilParaT \
        const REAL filter[3][3][3] = {\
          { {0.50/159,  0.0,  0.50/159},\
            {0.0,   0.0,  0.0},\
            {0.50/159,  0.0,  0.50/159}\
          },\
          { {0.51/159,  0.71/159, 0.91/159},\
            {1.21/159,  1.51/159, 1.21/159},\
            {0.91/159,  0.71/159, 0.51/159}\
          },\
          { {0.52/159,  0.0,  0.52/159},\
            {0.0,   0.0,  0.0},\
            {0.52/159,  0.0,  0.52/159}\
          }\
        };
    #endif
  #endif
  
  #define stencilParaList const REAL filter[halo*2+1][halo*2+1]
  #define stencilParaInput  filter
#endif


// #define R_PTR r_ptr[2*halo+1][INPUTREG_SIZE]

// template<class REAL, int RESULT_SIZE, int halo, int INPUTREG_SIZE=(RESULT_SIZE+2*halo)>
// __device__ void __forceinline__ computation(REAL result[RESULT_SIZE], 
//                                             REAL* sm_ptr, int sm_y_base, int sm_x_ind,int sm_width, 
//                                             REAL R_PTR,
//                                             int reg_base, 
//                                             stencilParaList
//                                             // const REAL west[6],const REAL east[6], 
//                                             // const REAL north[6],const REAL south[6],
//                                             // const REAL center 
//                                           )
// {
// #ifndef BOX
//   _Pragma("unroll")
//   for(int hl=0; hl<halo; hl++)
//   {
//     _Pragma("unroll")
//     for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
//     {
//       result[l_y]+=sm_ptr[(l_y+sm_y_base)*sm_width+sm_x_ind-1-hl]*west[hl];
//     }
//   }
//   _Pragma("unroll")
//   for(int hl=0; hl<halo; hl++)
//   {
//     _Pragma("unroll")
//     for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
//     {
//       result[l_y]+=r_ptr[reg_base+l_y-1-hl]*south[hl];
//     }
//   }
//   _Pragma("unroll")
//   for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
//   {
//     result[l_y]+=r_ptr[reg_base+l_y]*center;
//   }
//   _Pragma("unroll")
//   for(int hl=0; hl<halo; hl++)
//   {
//     _Pragma("unroll")
//     for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
//     {
//       result[l_y]+=r_ptr[reg_base+l_y+1+hl]*north[hl];
//     }
//   }
//   _Pragma("unroll")
//   for(int hl=0; hl<halo; hl++)
//   {
//     _Pragma("unroll")
//     for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
//     {
//       result[l_y]+=sm_ptr[(l_y+sm_y_base)*sm_width+sm_x_ind+1+hl]*east[hl];
//     }
//   }
// #else
//   _Pragma("unroll")\
//   for(int hl_y=-halo; hl_y<=halo; hl_y++)
//   {
//     _Pragma("unroll")
//     for(int hl_x=-halo; hl_x<=halo; hl_x++)
//     {
//       _Pragma("unroll")
//       for(int l_y=0; l_y<RESULT_SIZE ; l_y++)
//       {
//         result[l_y]+=filter[hl_y+halo][hl_x+halo]*r_ptr[hl_x+halo][hl_y+halo+l_y];
//       }
//     }
//   }
// #endif
// }



template<class REAL, int halo>
__global__ void kernel3d_restrict(REAL* input, REAL* output,
                                  int height, int width_y, int width_x); 


#define PERKS_DECLARE_INITIONIZATION_REFERENCE(_type,halo) \
    __global__ void kernel3d_restrict<_type,halo>(_type*,_type*,int,int,int);


// template<class REAL, int LOCAL_TILE_Y, int halo>
// __global__ void
// kernel_baseline (REAL*__restrict__ input, int width_y, int width_x, REAL*__restrict__ output);

// template<class REAL, int LOCAL_TILE_Y, int halo>
// __global__ void
// kernel_baseline_box (REAL* __restrict__ input, int width_y, int width_x, REAL* __restrict__ output);

// template<class REAL, int LOCAL_TILE_Y, int halo>
// __global__ void
// kernel_baseline_async (REAL*__restrict__ input, int width_y, int width_x, REAL*__restrict__ output);

// template<class REAL, int LOCAL_TILE_Y, int halo>
// __global__ void
// kernel_baseline_box_async (REAL* __restrict__ input, int width_y, int width_x, REAL* __restrict__ output);


// #define PERKS_DECLARE_INITIONIZATION_BASELINE(_type,tile,halo) \
//     __global__ void kernel_baseline<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__);

// #define PERKS_DECLARE_INITIONIZATION_BASELINE_BOX(_type,tile,halo) \
//     __global__ void kernel_baseline_box<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__);

// #define PERKS_DECLARE_INITIONIZATION_BASELINE_ASYNC(_type,tile,halo) \
//     __global__ void kernel_baseline_async<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__);

// #define PERKS_DECLARE_INITIONIZATION_BASELINE_BOX_ASYNC(_type,tile,halo) \
//     __global__ void kernel_baseline_box_async<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__);



// template<class REAL, int LOCAL_TILE_Y, int halo>
// __global__ void kernel_persistent_baseline(REAL *__restrict__  input, int width_y, int width_x, 
//   REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
//   int iteration);

//   template<class REAL, int LOCAL_TILE_Y, int halo>
// __global__ void kernel_persistent_baseline_box( REAL * __restrict__ input, int width_y, int width_x, 
//   REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
//   int iteration);

// template<class REAL, int LOCAL_TILE_Y, int halo>
// __global__ void kernel_persistent_baseline_async(REAL *__restrict__  input, int width_y, int width_x, 
//   REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
//   int iteration);

//   template<class REAL, int LOCAL_TILE_Y, int halo>
// __global__ void kernel_persistent_baseline_box_async( REAL * __restrict__ input, int width_y, int width_x, 
//   REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
//   int iteration);

// #define PERKS_DECLARE_INITIONIZATION_PBASELINE(_type,tile,halo) \
//     __global__ void kernel_persistent_baseline<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int );

// #define PERKS_DECLARE_INITIONIZATION_PBASELINE_BOX(_type,tile,halo) \
//     __global__ void kernel_persistent_baseline_box<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int );

// #define PERKS_DECLARE_INITIONIZATION_PBASELINE_ASYNC(_type,tile,halo) \
//     __global__ void kernel_persistent_baseline_async<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int );

// #define PERKS_DECLARE_INITIONIZATION_PBASELINE_BOX_ASYNC(_type,tile,halo) \
//     __global__ void kernel_persistent_baseline_box_async<_type,tile,halo>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int );

// template<class REAL, int LOCAL_TILE_Y, int halo,int reg_folder_y, bool UseSMCache>
// __global__ void kernel_general(REAL *__restrict__  input, int width_y, int width_x, 
//   REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
//   int iteration, int max_sm_flder);

// template<class REAL, int LOCAL_TILE_Y, int halo,int reg_folder_y, bool UseSMCache>
// __global__ void kernel_general_box( REAL * __restrict__ input, int width_y, int width_x, 
//   REAL *__restrict__  __var_4__,REAL *__restrict__  l2_cache, REAL *__restrict__  l2_cachetmp, 
//   int iteration, int max_sm_flder);

// #define PERKS_DECLARE_INITIONIZATION_GENERAL(_type,tile,halo,rf,usesm) \
//     __global__ void kernel_general<_type,tile,halo,rf,usesm>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int, int);

// #define PERKS_DECLARE_INITIONIZATION_GENERALBOX(_type,tile,halo,rf,usesm) \
//     __global__ void kernel_general_box<_type,tile,halo,rf,usesm>(_type*__restrict__,int,int,_type*__restrict__,_type*__restrict__,_type*__restrict__,int, int);
