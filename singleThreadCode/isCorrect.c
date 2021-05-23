#include <stdio.h>
#include <float.h>
#include "hpl-ai.h"

// Check final backward error.
int isCorrect(int n,double *A,int lda,double *x,double *b){
#ifdef HPL_CALL_MKL
    double norm_A = LAPACKE_dlange(LAPACK_COL_MAJOR,'I',n, n, A, lda);
    double norm_x = LAPACKE_dlange(LAPACK_COL_MAJOR,'I', n, 1, x, n);
    double norm_b = LAPACKE_dlange(LAPACK_COL_MAJOR,'I',n, 1, b, n);
    cblas_dgemv(CblasColMajor,CblasNoTrans,  n, n, 1.0, A, lda, x, 1, -1.0, b, 1);
    double threshold = 16.0;
    double eps = DBL_EPSILON / 2;
    double error = LAPACKE_dlange(LAPACK_COL_MAJOR,'I', n, 1, b, n) /
                               (norm_A * norm_x + norm_b) / n / eps;          
#else
    double norm_A = dlange('I', n, n, A, lda);
    double norm_x = dlange('I', n, 1, x, n);
    double norm_b = dlange('I', n, 1, b, n);
    dgemv('N', n, n, 1.0, A, lda, x, 1, -1.0, b, 1);
    double threshold = 16.0;
    double eps = DBL_EPSILON / 2;
    double error =
        dlange('I', n, 1, b, n) / (norm_A * norm_x + norm_b) / n / eps;
#endif
    printf("The following scaled residual check will be computed:\n");
    printf(
        "||Ax-b||_oo / ( eps * ( || x ||_oo * || A ||_oo + || b ||_oo ) * N "
        ")\n");
    printf("The relative machine precision (eps) is taken to be: %e\n", eps);
    printf("Computational tests pass if scaled residuals are less than %.1f\n",
           threshold);
    printf("||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)= %f ...", error);
    if (error < threshold) {
        printf("PASSED\n");
        return 1;
    } else {
        printf("FAILED\n");
        return 0;
    }
}