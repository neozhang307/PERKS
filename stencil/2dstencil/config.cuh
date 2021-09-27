#ifndef TILE_X
    #define TILE_X (256)
#endif 
#ifndef RTILE_Y
    #define RTILE_Y (8)
#endif

//minimal architecture is 600
#ifndef __CUDA_ARCH__
    #define PERKS_ARCH 000
#else
    #if __CUDA_ARCH__==800
        #define PERKS_ARCH 800
    #elif __CUDA_ARCH__==700
        #define PERKS_ARCH 700
    #elif __CUDA_ARCH__==600
        #define PERKS_ARCH 600
    #else
        #error "unsupport"
    #endif
#endif

#if defined(js2d5pt)
    #define HALO (1)
#elif defined(js2d9pt)
    #define HALO (2)
#elif defined(js2d13pt)
    #define HALO (2)
#elif defined(js2d17pt)
    #define HALO (3)
#elif defined(js2d21pt)
    #define HALO (4)
#elif defined(js2d25pt)
    #define HALO (5)
#elif defined(jb2d9pt)
    #define HALO (1)
#elif defined(jb2d25pt)
    #define HALO (2)
#endif

#ifndef Halo 
    #define Halo HALO
#endif