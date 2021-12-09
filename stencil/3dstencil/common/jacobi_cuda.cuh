#ifndef PERKS_CUDA_HEADER
#define PERKS_CUDA_HEADER
//template<class REAL>

//this is where the aimed implementation located
template<class REAL>
void j3d_iterative(REAL*, int, int, int, REAL*, int);

#define PERKS_DECLARE_INITIONIZATION_ITERATIVE(_type) \
    void j3d_iterative(_type*,int,int, int,_type*, int);

#endif
