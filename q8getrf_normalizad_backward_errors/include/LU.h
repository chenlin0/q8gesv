#ifndef _LU_H_
#define _LU_H_
#include "hpl.h"
#include<stdlib.h>

#ifdef __cplusplus 
extern "C" {
#endif
typedef struct HPL_LU_t
{
    HPL_T_pmat mat;
    HPL_T_grid grid;
    HPL_T_palg algo;
    HPL_T_test test;
    int N;
    int NB;
    int P;
    int Q;
    void* vptr;
    double *gflops;
    int nTest;
}HPL_LU;
//初始化，分配内存
int LU_init(HPL_LU *lu); 
//读取LU.dat，进行求Ax=b   
int LU_solve(HPL_LU *lu);
//释放内存
int LU_exit(HPL_LU *lu);
void convert_double_to_float(_ITER_D_TYPE_ *src, int ldsrc, _D_TYPE_ *dst,
                             int lddst, int m, int n);
void convert_float_to_double(_D_TYPE_ *src, int ldsrc, _ITER_D_TYPE_ *dst,
                             int lddst, int m, int n);
void copyDouble2Double(_ITER_D_TYPE_ *src, int ldsrc, _ITER_D_TYPE_ *dst,
                             int lddst, int m, int n);
void HPL_gmres(int n, double* A, int lda, double* x, double* b, double* LU,
           int ldlu,  int max_it, double tol);     

#ifdef __cplusplus 
}
#endif                                                                                   
#endif