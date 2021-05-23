#include "hpl-ai.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include<string.h>

int main(int argc, char* argv[]) {
    mkl_set_num_threads(1);
    int n = 2000;      // matrix size
    int max_it = 50;  // maximum number of iterations in GMRES
    if (argc >= 2) {
        n = atoi(argv[1]);
    }
    if (argc >= 3) {
        max_it = atoi(argv[2]);
    }
    if (max_it >= n) {
        max_it = n - 1;
    }

    double t0,time_total;

    int lda = (n + 16 - 1) / 16 * 16;  // round up to multiple of 16
    unsigned long long iseedA = 1;      // RNG seed of matrix A
    unsigned long long iseedB = 100;      // RNG seed of vector b
    double* A = (double*)malloc(lda * n * sizeof(double));

    double* b = (double*)malloc(n * sizeof(double));
    double* x = (double*)malloc(n * sizeof(double));
    int* ipiv = (int*)malloc(n*sizeof(int));



    printf(
        "======================================================================"
        "==========\n");
    printf(
        "                        HPL-AI Mixed-Precision Benchmark              "
        "          \n");
    printf(
        "       Written by Yaohung Mike Tsai, Innovative Computing Laboratory, "
        "UTK       \n");
    printf(
        "======================================================================"
        "==========\n");
    printf("\n");
    printf(
        "This is a reference implementation with the matrix generator, an "
        "example\n");
    printf(
        "mixed-precision solver with LU factorization in single and GMRES in "
        "double,\n");
    printf("as well as the scaled residual check.\n");
    printf(
        "Please visit http://www.icl.utk.edu/research/hpl-ai for more "
        "details.\n");
    printf("\n");
    printf("=====================================\n");
    printf("input size: n = %d\n",n);
    printf("dgesv: \n");
    int nTest = 20000/n;
    nTest = (nTest<10)?10:nTest;
    double ops = 2.0 / 3.0 * n * n * n + 3.0 / 2.0 * n * n;

    time_total = 0;
    memset(x,0,n*sizeof(double));
    for(int i=0;i<nTest;i++){
        matgen(A, lda, n, iseedA);
        vecgen(b, n, iseedB);
        t0 = get_wtime();
        LAPACKE_dgesv(LAPACK_COL_MAJOR,n,1,A,lda,ipiv,b,n);
        time_total += get_wtime() - t0;
    }
    time_total /= nTest;

    memcpy(x,b,n*sizeof(double));
    printf("dgesv      : %12f GFLOPs\n",
           1e-9 * ops / time_total);
    matgen(A, lda, n, iseedA);
    vecgen(b, n, iseedB);
    isCorrect(n,A,lda,x,b);

    printf("________________________________\n");
    printf("HPL AI Benchmark: \n");
    time_total = 0;
    memset(x,0,n*sizeof(double));
    for(int i=0;i<nTest;i++){
        matgen(A, lda, n, iseedA);
        vecgen(b, n, iseedB);
        t0 = get_wtime();
        HPL_AI_gesv(n,A,lda,b,x,max_it);
        time_total += get_wtime() - t0;
    }
    time_total /= nTest;
    printf("HPL-AI Mixed-Precision Benchmark: %12f GFLOPs\n",
           1e-9 * ops / time_total);
    matgen(A, lda, n, iseedA);
    vecgen(b, n, iseedB);
    isCorrect(n,A,lda,x,b);
    printf("________________________________\n");
    /*
    printf("dgesvxx: \n");
    time_total = 0;
    memset(x,0,n*sizeof(double));
    ipiv = (int*)malloc(n*sizeof(int));
    char fact = 'N';
    char trans = 'N';
    double *Af = (double*)malloc(lda * n * sizeof(double));
    double *r = (double*)malloc( n * sizeof(double)); 
    double *c = (double*)malloc( n * sizeof(double)); 
    for(int i=0;i<n;i++){
        r[i] = c[i] = 1;
    }
    double err_bnds_norm,err_bnds_cmp;
    double params=1;
    err_bnds_norm = 1;
    err_bnds_cmp = 1;
    double *rcond = (double*)malloc( n * sizeof(double)); 
    double *rpvgrw = (double*)malloc( n * sizeof(double)); 
    double *berr = (double*)malloc( n * sizeof(double)); 
    char equed;
    for(int i=0;i<nTest;i++){
        matgen(A, lda, n, iseedA);
        vecgen(b, n, iseedB);
        t0 = get_wtime();
        LAPACKE_dgesvxx(LAPACK_COL_MAJOR,fact,trans,n,1,A,lda,Af,lda,ipiv,
        &equed,r,c,b,n,x,n,rcond,rpvgrw,berr,1,&err_bnds_norm,&err_bnds_cmp,0,&params);
        HPL_AI_gesv(n,A,lda,b,x,max_it);
        time_total += get_wtime() - t0;
    }
    time_total /= nTest;
    printf("dgesvxx: %12f GFLOPs\n",
           1e-9 * ops / time_total);
    matgen(A, lda, n, iseedA);
    vecgen(b, n, iseedB);
    isCorrect(n,A,lda,x,b);
    free(Af);
    free(r);
    free(c);
    free(rcond);
    free(rpvgrw);
    free(berr);
    */
    printf("________________________________\n");
    printf("dsgesv: \n");
    time_total = 0;
    memset(x,0,n*sizeof(double));
    int iter = max_it;
    for(int i=0;i<nTest;i++){
        matgen(A, lda, n, iseedA);
        vecgen(b, n, iseedB);
        t0 = get_wtime();
        LAPACKE_dsgesv(LAPACK_COL_MAJOR,n,1,A,lda,ipiv,b,n,x,n,&iter);
        time_total += get_wtime() - t0;
    }
    time_total /= nTest;
    printf("dsgesv: %12f GFLOPs\n",
           1e-9 * ops / time_total);
    matgen(A, lda, n, iseedA);
    vecgen(b, n, iseedB);
    isCorrect(n,A,lda,x,b);
    printf("________________________________\n");
    printf("q8gesv: \n");
    time_total = 0;
    memset(x,0,n*sizeof(double));
    for(int i=0;i<nTest;i++){
        matgen(A, lda, n, iseedA);
        vecgen(b, n, iseedB);
        t0 = get_wtime();
        q8gesv(n,A,lda,b,x,max_it);
        time_total += get_wtime() - t0;
    }
    time_total /= nTest;
    printf("q8gesv: %12f GFLOPs\n",
           1e-9 * ops / time_total);
    matgen(A, lda, n, iseedA);
    vecgen(b, n, iseedB);
    isCorrect(n,A,lda,x,b);
    printf("=====================================\n");

    free(A);
    free(x);
    free(b);
    free(ipiv);
    return 0;
}
