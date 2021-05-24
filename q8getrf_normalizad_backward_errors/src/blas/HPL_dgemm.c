/* 
 * -- High Performance Computing Linpack Benchmark (HPL)                
 *    HPL - 2.3 - December 2, 2018                          
 *    Antoine P. Petitet                                                
 *    University of Tennessee, Knoxville                                
 *    Innovative Computing Laboratory                                 
 *    (C) Copyright 2000-2008 All Rights Reserved                       
 *                                                                      
 * -- Copyright notice and Licensing terms:                             
 *                                                                      
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:                                                             
 *                                                                      
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.        
 *                                                                      
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution. 
 *                                                                      
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:                 
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.             
 *                                                                      
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.                                                          
 *                                                                      
 * -- Disclaimer:                                                       
 *                                                                      
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 * ---------------------------------------------------------------------
 */ 
/*
 * Include files
 */
//#include "hpl.h"
#include<stdint.h>
#include<math.h>
#include<mkl.h>
#include<string.h>
#include<stdio.h>
#include"hpl_ai.h"

enum HPL_ORDER
{  HplRowMajor = 101,  HplColumnMajor  = 102 };
enum HPL_TRANS
{  HplNoTrans  = 111,  HplTrans        = 112,  HplConjTrans    = 113 };
enum HPL_UPLO
{  HplUpper    = 121,  HplLower        = 122 };
enum HPL_DIAG
{  HplNonUnit  = 131,  HplUnit         = 132 };
enum HPL_SIDE
{  HplLeft     = 141,  HplRight        = 142 }; 


void rowMaxValue(float* A,int m,int n,int ld,float *maxValue){
   int i,j;
   float temp;
   for(i=0;i<m;i++){
      maxValue[i] = fabs(*A);
      A++;
   }
   A += ld - m;
   for(j=1;j<n;++j){
      for(i=0;i<m;++i){
         temp = fabs(*A);
         if( temp > maxValue[i]){
            maxValue[i] = temp;
         }
         A++;
      }
      A += ld - m;
   }
   
} 
void colMaxValue(float* A,int m,int n,int ld,float *maxValue){
   int i,j;
   float temp;
   for(j=0;j<n;++j){
      maxValue[j] = fabs(*A);
      for(i=0;i<m;++i){
         temp = fabs(*A);
         if( temp > maxValue[j]){
            maxValue[j] = temp;
         }
         A++;
      }
      A += ld - m;
   }
   
} 
void s8RowQuant(float *A,int m,int n,int ld,float *scalar,int8_t *a){
   float *Aptr = A;
   int8_t *aptr = a;
   int i,j;   
   rowMaxValue(A,m,n,ld,scalar);
   for(i=0;i<m;i++){
      scalar[i] /= 127;
   }
   for(j=0;j<n;++j){
      for(i=0;i<m;++i){
         *aptr = (int8_t)((*Aptr)/scalar[i]);
         Aptr++;
         aptr++;
      }
      Aptr += ld - m;
   }
}
void u8ColQuant(float *A,int m,int n,int ld,float *scalar,int8_t *a){
   float *Aptr = A;
   int8_t *aptr = a;
   int i,j;
   colMaxValue(A,m,n,ld,scalar);
   for(j=0;j<n;j++){
      scalar[j] /= 127;
   }
   for(j=0;j<n;++j){
      for(i=0;i<m;++i){
         *aptr = (uint8_t)((*Aptr)/scalar[j]+127);
         Aptr ++;
         aptr++;
      }
      Aptr += ld - m;
   }
}
void q8gemm(int m,int n,int k,float *A,int lda,float *B,int ldb,\
float *C,int ldc){

   int i,j,l,ii,jj,i0,j0,k0;
   float rowTemp,colTemp;
   MKL_INT8 *a = (MKL_INT8 *)mkl_malloc(m*k*sizeof( MKL_INT8  ),64);
   MKL_INT8 *b = (MKL_INT8 *)mkl_malloc(k*n*sizeof( MKL_INT8  ),64);
   MKL_INT32 *c = (MKL_INT32 *)mkl_malloc(MC*NC*sizeof( MKL_INT32 ),64);
   float *Sa = (float*) mkl_malloc(m*sizeof(float),64);
   float *Sb = (float*) mkl_malloc(n*sizeof(float),64);

   s8RowQuant(A,m,k,lda,Sa,a);
   u8ColQuant(B,k,n,ldb,Sb,b);

   

   MKL_INT32 co = 0;
   int mb = (m+MC-1) / MC;
   int nb = (n+NC-1) / NC;
   int kb = (k+KC-1) / KC;
   int _mc = m % MC;
   int _nc = n % NC;
   int _kc = k % KC;
   int mc,nc,kc;
   for(l=0;l<kb;++l){
      kc = (l!=kb-1 || _kc ==0)? KC: _kc;
      k0 = l*kc;
      for(i=0;i<mb;++i){
         mc = (i!=mb-1 || _mc==0)? MC : _mc;
         i0 = i*mc;
         for(j=0;j<nb;++j){
            nc = (j!=nb-1|| _nc==0)? NC : _nc;
            j0 = j*nc;
            cblas_gemm_s8u8s32(CblasColMajor,CblasNoTrans,CblasNoTrans,CblasFixOffset,mc,nc,kc,1,a+i0+k0*m,m,0,b+k0+j0*k,k,-127,0,c,MC,&co);
            float *Cptr = C+i0+j0*ldc;
            int32_t *cptr = c;
            for(jj=0;jj<nc;++jj){
               colTemp = Sb[j0+jj];
               for(ii=0;ii<mc;++ii){
                  *Cptr -= Sa[i0+ii]*colTemp*(*cptr);
                  Cptr++;
                  cptr++;
               }
               Cptr += ldc - mc;
               cptr += MC - mc;

            }

         }
      }
   }
   mkl_free(a);
   mkl_free(b);
   mkl_free(c);
   mkl_free(Sa);
   mkl_free(Sb);
}
#ifndef HPL_dgemm



#ifdef STDC_HEADERS
void HPL_dgemm
(
   const enum HPL_ORDER             ORDER,
   const enum HPL_TRANS             TRANSA,
   const enum HPL_TRANS             TRANSB,
   const int                        M,
   const int                        N,
   const int                        K,
   const _D_TYPE_                     ALPHA,
   const _D_TYPE_ *                   A,
   const int                        LDA,
   const _D_TYPE_ *                   B,
   const int                        LDB,
   const _D_TYPE_                     BETA,
   _D_TYPE_ *                         C,
   const int                        LDC
)
#else
void HPL_dgemm
( ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
   const enum HPL_ORDER             ORDER;
   const enum HPL_TRANS             TRANSA;
   const enum HPL_TRANS             TRANSB;
   const int                        M;
   const int                        N;
   const int                        K;
   const _D_TYPE_                     ALPHA;
   const _D_TYPE_ *                   A;
   const int                        LDA;
   const _D_TYPE_ *                   B;
   const int                        LDB;
   const _D_TYPE_                     BETA;
   _D_TYPE_ *                         C;
   const int                        LDC;
#endif
{
/* 
 * Purpose
 * =======
 *
 * HPL_dgemm performs one of the matrix-matrix operations
 *  
 *     C := alpha * op( A ) * op( B ) + beta * C
 *  
 *  where op( X ) is one of
 *  
 *     op( X ) = X   or   op( X ) = X^T.
 *  
 * Alpha and beta are scalars,  and A,  B and C are matrices, with op(A)
 * an m by k matrix, op(B) a k by n matrix and  C an m by n matrix.
 *
 * Arguments
 * =========
 *
 * ORDER   (local input)                 const enum HPL_ORDER
 *         On entry, ORDER  specifies the storage format of the operands
 *         as follows:                                                  
 *            ORDER = HplRowMajor,                                      
 *            ORDER = HplColumnMajor.                                   
 *
 * TRANSA  (local input)                 const enum HPL_TRANS
 *         On entry, TRANSA  specifies the form of  op(A)  to be used in
 *         the matrix-matrix operation follows:                         
 *            TRANSA==HplNoTrans    : op( A ) = A,                     
 *            TRANSA==HplTrans      : op( A ) = A^T,                   
 *            TRANSA==HplConjTrans  : op( A ) = A^T.                   
 *
 * TRANSB  (local input)                 const enum HPL_TRANS
 *         On entry, TRANSB  specifies the form of  op(B)  to be used in
 *         the matrix-matrix operation follows:                         
 *            TRANSB==HplNoTrans    : op( B ) = B,                     
 *            TRANSB==HplTrans      : op( B ) = B^T,                   
 *            TRANSB==HplConjTrans  : op( B ) = B^T.                   
 *
 * M       (local input)                 const int
 *         On entry,  M  specifies  the  number  of rows  of the  matrix
 *         op(A)  and  of  the  matrix  C.  M  must  be  at least  zero.
 *
 * N       (local input)                 const int
 *         On entry,  N  specifies  the number  of columns of the matrix
 *         op(B)  and  the number of columns of the matrix  C. N must be
 *         at least zero.
 *
 * K       (local input)                 const int
 *         On entry,  K  specifies  the  number of columns of the matrix
 *         op(A) and the number of rows of the matrix op(B).  K  must be
 *         be at least  zero.
 *
 * ALPHA   (local input)                 const _D_TYPE_
 *         On entry, ALPHA specifies the scalar alpha.   When  ALPHA  is
 *         supplied  as  zero  then the elements of the matrices A and B
 *         need not be set on input.
 *
 * A       (local input)                 const _D_TYPE_ *
 *         On entry,  A  is an array of dimension (LDA,ka),  where ka is
 *         k  when   TRANSA==HplNoTrans,  and  is  m  otherwise.  Before
 *         entry  with  TRANSA==HplNoTrans, the  leading  m by k part of
 *         the array  A must contain the matrix A, otherwise the leading
 *         k  by  m  part of the array  A  must  contain the  matrix  A.
 *
 * LDA     (local input)                 const int
 *         On entry, LDA  specifies the first dimension of A as declared
 *         in the  calling (sub) program. When  TRANSA==HplNoTrans  then
 *         LDA must be at least max(1,m), otherwise LDA must be at least
 *         max(1,k).
 *
 * B       (local input)                 const _D_TYPE_ *
 *         On entry, B is an array of dimension (LDB,kb),  where  kb  is
 *         n   when  TRANSB==HplNoTrans, and  is  k  otherwise.   Before
 *         entry with TRANSB==HplNoTrans,  the  leading  k by n  part of
 *         the array  B must contain the matrix B, otherwise the leading
 *         n  by  k  part of the array  B  must  contain  the matrix  B.
 *
 * LDB     (local input)                 const int
 *         On entry, LDB  specifies the first dimension of B as declared
 *         in the  calling (sub) program. When  TRANSB==HplNoTrans  then
 *         LDB must be at least max(1,k), otherwise LDB must be at least
 *         max(1,n).
 *
 * BETA    (local input)                 const _D_TYPE_
 *         On entry,  BETA  specifies the scalar  beta.   When  BETA  is
 *         supplied  as  zero  then  the  elements of the matrix C  need
 *         not be set on input.
 *
 * C       (local input/output)          _D_TYPE_ *
 *         On entry,  C  is an array of dimension (LDC,n). Before entry,
 *         the  leading m by n part  of  the  array  C  must contain the
 *         matrix C,  except when beta is zero, in which case C need not
 *         be set on entry. On exit, the array  C  is overwritten by the
 *         m by n  matrix ( alpha*op( A )*op( B ) + beta*C ).
 *
 * LDC     (local input)                 const int
 *         On entry, LDC  specifies the first dimension of C as declared
 *         in  the   calling  (sub)  program.   LDC  must  be  at  least
 *         max(1,m).
 *
 * ---------------------------------------------------------------------
 */ 
#ifdef HPL_CALL_CBLAS
#ifdef HPL_AI_DOUBLE
   //cblas_dgemm( ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB,
   //             BETA, C, LDC );


#endif
#ifdef HPL_AI_FLOAT
   if(K==valueKq8){
      q8gemm(M,N,K,A,LDA,B,LDB,C,LDC);
   }
   else{
      cblas_sgemm( ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB,
                BETA, C, LDC );
   }

   
   
#endif
#ifdef HPL_AI_COMPLEX_FLOAT
   cblas_cgemm( ORDER, TRANSA, TRANSB, M, N, K, (void*)(&ALPHA), (void*)A, 
   LDA, (void*)B, LDB,(void*)(&BETA), (void*)C, LDC );
#endif
#ifdef HPL_AI_COMPLEX_DOUBLE
   cblas_zgemm( ORDER, TRANSA, TRANSB, M, N, K, (void*)(&ALPHA), (void*)A, 
   LDA, (void*)B, LDB,(void*)(&BETA), (void*)C, LDC );
#endif

#endif

/*
 * End of HPL_dgemm
 */
}

#endif
