#include<stdint.h>
#include<math.h>
#include<mkl.h>
#include<string.h>
#include"hpl-ai.h"
#include<stdio.h>
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
void s8RowQuant(float *A,int m,int n,int ld,float *scalar,MKL_INT8 *a){
   float *Aptr = A;
   MKL_INT8 *aptr = a;
   int i,j;   
   rowMaxValue(A,m,n,ld,scalar);
   for(i=0;i<m;i++){
      scalar[i] /= 127;
   }
   for(j=0;j<n;++j){
      for(i=0;i<m;++i){
         *aptr = (MKL_INT8)((*Aptr)/scalar[i]);
         Aptr++;
         aptr++;
      }
      Aptr += ld - m;
   }
}
void u8ColQuant(float *A,int m,int n,int ld,float *scalar,MKL_INT8 *a){
   float *Aptr = A;
   MKL_INT8 *aptr = a;
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
void s8gemm(int m,int n,int k,float *A,int lda,float *B,int ldb,\
float *C,int ldc){
   int i,j;
   MKL_INT8 *a = (MKL_INT8 *)  mkl_calloc(m*k, sizeof( MKL_INT8  ), 64);
   MKL_INT8 *b = (MKL_INT8 *)  mkl_calloc(k*n, sizeof( MKL_INT8  ), 64);
   MKL_INT32 *c = (MKL_INT32 *) mkl_calloc(m*n, sizeof( MKL_INT32 ), 64);
   float *Sa = (float*) mkl_calloc(m,sizeof(float),64);
   float *Sb = (float*) mkl_calloc(n,sizeof(float),64);
   s8RowQuant(A,m,k,lda,Sa,a);
   u8ColQuant(B,k,n,ldb,Sb,b);
   MKL_INT32 co = 0;
   memset(c,0,m*n*sizeof(int32_t));
   for(j=0;j<n;++j){
      for(i=0;i<m;++i){
         for(int l=0;l<k;++l){
            c[j*m+i] += ((int32_t)(a[i+l*m]))*((int32_t)(b[l+j*k])-127);
         }
      }
   }   
   float *Cptr = C;
   int32_t *cptr = c;
   for(j=0;j<n;++j){
      for(i=0;i<m;++i){
         *Cptr -= Sa[i]*Sb[j]*(*cptr);
         Cptr++;
         cptr++;
      }
      Cptr += ldc - m;
   }
   mkl_free(a);
   mkl_free(b);
   mkl_free(c);
   mkl_free(Sa);
   mkl_free(Sb);
}
void q8gemm_old(int m,int n,int k,float *A,int lda,float *B,int ldb,\
float *C,int ldc){

   int i,j;
   float rowTemp,colTemp;
   MKL_INT8 *a = (MKL_INT8 *)mkl_malloc(m*k*sizeof( MKL_INT8  ),64);
   MKL_INT8 *b = (MKL_INT8 *)mkl_malloc(k*n*sizeof( MKL_INT8  ),64);
   MKL_INT32 *c = (MKL_INT32 *)mkl_malloc(m*n*sizeof( MKL_INT32 ),64);
   float *Sa = (float*) mkl_malloc(m*sizeof(float),64);
   float *Sb = (float*) mkl_malloc(n*sizeof(float),64);

   s8RowQuant(A,m,k,lda,Sa,a);


   u8ColQuant(B,k,n,ldb,Sb,b);

   

   MKL_INT32 co = 0;
   cblas_gemm_s8u8s32(CblasColMajor,CblasNoTrans,CblasNoTrans,CblasFixOffset,\
   m,n,k,1,a,m,0,b,k,-127,0,c,m,&co);


   float *Cptr = C;
   int32_t *cptr = c;
   for(j=0;j<n;++j){
      colTemp = Sb[j];
      for(i=0;i<m;++i){
         *Cptr -= Sa[i]*colTemp*(*cptr);
         Cptr++;
         cptr++;
      }
      Cptr += ldc - m;
   }

   mkl_free(a);
   mkl_free(b);
   mkl_free(c);
   mkl_free(Sa);
   mkl_free(Sb);

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
   //GEBP
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
