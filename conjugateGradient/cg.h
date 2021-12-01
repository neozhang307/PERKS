#define MAXITER (10000)

void cpuConjugateGrad(int *I, int *J, float *val, float *x, float *Ax, float *p,
                      float *r, int nnz, int N, float tol);