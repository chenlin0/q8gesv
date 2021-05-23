#include <stdio.h>
#include <float.h>
#include "hpl-ai.h"

#define A(i, j) *HPLAI_INDEX2D(A, (i), (j), lda)

void q8getrf_nopiv(int m, int n, float *A, int lda) {

    int j;
    const int nb = 512;
    int jb = nb;

    // Use unblock code.
    if (nb > m || nb > n) {
        sgetrf2_nopiv2(m, n, A, lda);
        return;
    }

    int min_mn = m < n ? m : n;

    for (j = 0; j < min_mn; j += nb) {
        if (min_mn - j < nb) {
            jb = min_mn - j;
        }

        // Factor panel
        sgetrf2_nopiv2(m - j, jb, &A(j, j), lda);

        if (j + jb < n) {
	#ifdef  HPL_CALL_MKL
            cblas_strsm (CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasUnit ,jb, n - j - jb, 1.0, &A(j, j), lda,
                  &A(j, j + jb), lda);
	#else
            strsm('L', 'L', 'N', 'U', jb, n - j - jb, 1.0, &A(j, j), lda,
                  &A(j, j + jb), lda);
	#endif
            if (j + jb < m) {

                q8gemm( m - j - jb, n - j - jb, jb, &A(j + jb, j),
                      lda, &A(j, j + jb), lda, &A(j + jb, j + jb), lda);



            }
        }
    }
}

void sgetrf2_nopiv2(int m, int n, float *A, int lda) {

    int i;

    if (m <= 1 || n == 0) {
        return;
    }

    if (n == 1) {
        for (i = 1; i < m; i++) {
            A(i, 0) /= A(0, 0);
        }
    } else {  // Use recursive code

        int n1 = (m > n ? n : m) / 2;
        int n2 = n - n1;

        sgetrf2_nopiv2(m, n1, A, lda);
	#ifdef  HPL_CALL_MKL
        cblas_strsm (CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasUnit, n1, n2, 1.0, A, lda, &A(0, n1), lda);
	#else
        strsm('L', 'L', 'N', 'U', n1, n2, 1.0, A, lda, &A(0, n1), lda);
	#endif

    	#ifdef  HPL_CALL_MKL
        cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m - n1, n2, n1, -1.0, &A(n1, 0), lda, &A(0, n1), lda,
              1.0, &A(n1, n1), lda);
    	#else
        sgemm('N', 'N', m - n1, n2, n1, -1.0, &A(n1, 0), lda, &A(0, n1), lda,
              1.0, &A(n1, n1), lda);
    	#endif
        sgetrf2_nopiv2(m - n1, n2, &A(n1, n1), lda);
    }
    return;
}
void q8gesv(int n,double *A,int lda,double *b,double *x,int max_it){
    float* sA = (float*)malloc(lda * n * sizeof(float));
    float* sb = (float*)malloc(n * sizeof(float));
    double* LU = (double*)malloc(lda * n * sizeof(double));
    // Convert A and b to single
    convert_double_to_float(A, lda, sA, lda, n, n);
    convert_double_to_float(b, n, sb, n, n, 1);
    q8getrf_nopiv(n, n, sA, lda);
    // Forward and backward substitution.
#ifdef HPL_CALL_MKL
    cblas_strsm (CblasColMajor,CblasLeft,CblasLower,CblasNoTrans,CblasUnit ,  n, 1, 1.0, sA, lda, sb, n);   
    cblas_strsm (CblasColMajor,CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit,  n, 1, 1.0, sA, lda, sb, n);   
#else
    strsm('L', 'L', 'N', 'U', n, 1, 1.0, sA, lda, sb, n);
    strsm('L', 'U', 'N', 'N', n, 1, 1.0, sA, lda, sb, n);
#endif
    // Convert result back to double.
    convert_float_to_double(sA, lda, LU, lda, n, n);
    convert_float_to_double(sb, n, x, n, n, 1);
    // Using GMRES without restart.
    // GMRES is checking preconditioned residual so the tolerance is smaller.
    double tol = DBL_EPSILON / 2.0 / ((double)n / 4.0);
    gmres(n, A, lda, x, b, LU, lda, max_it, tol);
    free(sA);
    free(sb);
    free(LU);
}