#ifndef HPL_AI_H
#define HPL_AI_H
#include"mpi.h"
#include<complex.h>
#include <math.h>
#include<float.h>
#define MC 848
#define NC 848
#define KC 512
//when K is equal to valueKq8, HPL_dgemm call q8gemm
//otherwise, HPL_dgemm call cblas_sgemm
#define valueKq8 512
#define _REAL_TYPE_ double

#ifdef HPL_AI_FLOAT
#undef HPL_AI_DOUBLE
#undef HPL_AI_COMPLEX_FLOAT
#undef HPL_AI_COMPLEX_DOUBLE
#define _D_TYPE_ float
#define _M_TYPE_ MPI_FLOAT
#define  _ITER_D_TYPE_ double
#define  _ITER_M_TYPE_ MPI_DOUBLE
#define realPart(a) (a) 
#define imagPart(a) 0 
inline void convert_double_to_float(double *src, int ldsrc, float *dst,
                             int lddst, int m, int n) {
    int i, j;
    for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i){
            *dst = (float)(*src);
            dst++;
            src++;
        }
        dst += lddst - m;
        src += ldsrc - m;
    }
    return;
}
inline void convert_float_to_double(float *src, int ldsrc, double *dst,
                             int lddst, int m, int n) {
    int i, j;
    for (j = 0; j < n; ++j) {
        for (i = 0; i < m; ++i){
            *dst = (double)(*src);
            dst++;
            src++;
        }
        dst += lddst - m;
        src += ldsrc - m;
    }
    return;
}

#endif

#ifdef HPL_AI_DOUBLE
#undef HPL_AI_FLOAT
#undef HPL_AI_COMPLEX_FLOAT
#undef HPL_AI_COMPLEX_DOUBLE
#define _D_TYPE_ double
#define _M_TYPE_ MPI_DOUBLE
#define  _ITER_D_TYPE_ double
#define  _ITER_M_TYPE_ MPI_DOUBLE
#define realPart(a) (a) 
#define imagPart(a) 0 
#endif

#ifdef HPL_AI_COMPLEX_FLOAT
#undef HPL_AI_DOUBLE
#undef HPL_AI_FLOAT
#undef HPL_AI_COMPLEX_DOUBLE
#define _D_TYPE_ float _Complex
#define _M_TYPE_ MPI_C_FLOAT_COMPLEX
#define  _ITER_D_TYPE_ double _Complex
#define  _ITER_M_TYPE_ MPI_C_DOUBLE_COMPLEX 
#define realPart(a) (creal((_D_TYPE_)(a))) 
#define imagPart(a) (cimag((_D_TYPE_)(a))) 
#endif

#ifdef HPL_AI_COMPLEX_DOUBLE
#undef HPL_AI_DOUBLE
#undef HPL_AI_FLOAT
#undef HPL_AI_COMPLEX_FLOAT
#define _D_TYPE_ double _Complex
#define _M_TYPE_ MPI_C_DOUBLE_COMPLEX 
#define  _ITER_D_TYPE_ double _Complex
#define  _ITER_M_TYPE_ MPI_C_DOUBLE_COMPLEX 
#define realPart(a) (creal((_D_TYPE_)(a))) 
#define imagPart(a) (cimag((_D_TYPE_)(a))) 
#endif


#endif