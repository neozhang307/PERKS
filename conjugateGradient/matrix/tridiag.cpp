
#include "./tridiag.h"

#include <stdio.h>
#include <algorithm>
// includes, system
/* genTridiag: generate a random tridiagonal symmetric matrix */
template <
    typename ValueT,
    typename OffsetT>
void genTridiag(OffsetT *I, OffsetT *J, ValueT *val, OffsetT N, OffsetT nz) {
  I[0] = 0, J[0] = 0, J[1] = 1;
  val[0] = static_cast<ValueT>(rand()) / RAND_MAX + 10.0f;
  val[1] = static_cast<ValueT>(rand()) / RAND_MAX;
  int start;

  for (int i = 1; i < N; i++) {
    if (i > 1) {
      I[i] = I[i - 1] + 3;
    } else {
      I[1] = 2;
    }

    start = (i - 1) * 3 + 2;
    J[start] = i - 1;
    J[start + 1] = i;

    if (i < N - 1) {
      J[start + 2] = i + 1;
    }

    val[start] = val[start - 1];
    val[start + 1] = static_cast<ValueT>(rand()) / RAND_MAX + 10.0f;

    if (i < N - 1) {
      val[start + 2] = static_cast<ValueT>(rand()) / RAND_MAX;
    }
  }

  I[N] = nz;
}

// I - contains location of the given non-zero element in the row of the matrix
// J - contains location of the given non-zero element in the column of the
// matrix val - contains values of the given non-zero elements of the matrix
// inputVecX - input vector to be multiplied
// outputVecY - resultant vector


template void genTridiag(int *I, int *J, float *val, int N, int nz);
template void genTridiag(int *I, int *J, double *val, int N, int nz);