#include "cg.h"

void cpuSpMV(int *I, int *J, float *val, int nnz, int num_rows, float alpha,
             float *inputVecX, float *outputVecY) {
  for (int i = 0; i < num_rows; i++) {
    int num_elems_this_row = I[i + 1] - I[i];

    float output = 0.0;
    for (int j = 0; j < num_elems_this_row; j++) {
      output += alpha * val[I[i] + j] * inputVecX[J[I[i] + j]];
    }
    outputVecY[i] = output;
  }

  return;
}

double dotProduct(float *vecA, float *vecB, int size) {
  double result = 0.0;

  for (int i = 0; i < size; i++) {
    result = result + (vecA[i] * vecB[i]);
  }

  return result;
}

void scaleVector(float *vec, float alpha, int size) {
  for (int i = 0; i < size; i++) {
    vec[i] = alpha * vec[i];
  }
}

void saxpy(float *x, float *y, float a, int size) {
  for (int i = 0; i < size; i++) {
    y[i] = a * x[i] + y[i];
  }
}

void cpuConjugateGrad(int *I, int *J, float *val, float *x, float *Ax, float *p,
                      float *r, int nnz, int N, float tol) {
  int max_iter = MAXITER;

  float alpha = 1.0;
  float alpham1 = -1.0;
  float r0 = 0.0, b, a, na;

  cpuSpMV(I, J, val, nnz, N, alpha, x, Ax);
  saxpy(Ax, r, alpham1, N);

  float r1 = dotProduct(r, r, N);

  int k = 1;

  // while (r1 > tol * tol && k <= max_iter) {
  while ( k <= max_iter) {
    if (k > 1) {
      b = r1 / r0;
      scaleVector(p, b, N);

      saxpy(r, p, alpha, N);
    } else {
      for (int i = 0; i < N; i++) p[i] = r[i];
    }
// #ifndef NMUL
    cpuSpMV(I, J, val, nnz, N, alpha, p, Ax);
// #endif
    float dot = dotProduct(p, Ax, N);
    a = r1 / dot;

    saxpy(p, x, a, N);
    na = -a;
    saxpy(Ax, r, na, N);

    r0 = r1;
    r1 = dotProduct(r, r, N);

    // printf("\nCPU code iteration = %3d, residual = %e\n", k, sqrt(r1));
    k++;
  }
}