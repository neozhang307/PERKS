#ifndef PERKS_CUDA_HEADER
#define PERKS_CUDA_HEADER
//template<class REAL>

//this is where the aimed implementation located
template<class REAL>
void jacobi_iterative(REAL*, int, int, REAL*, int, int, int, bool,bool,bool,int);

#define PERKS_DECLARE_INITIONIZATION_ITERATIVE(_type) \
    void jacobi_iterative(_type*,int,int,_type*, int, int, int, bool,bool,bool,int);

template<int halo, bool isstar, int arch, class REAL>
int getMinWidthY(int , int );

template<class REAL>int getMinWidthY(int , int , int , bool );
template<class REAL>int getMinWidthY(int , int , int );
template<class REAL>int getMinWidthY(int , int );

#endif
