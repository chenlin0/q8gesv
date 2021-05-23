#include<stdio.h>
#include<stdlib.h>
#include"hpl-ai.h"
#include<time.h>
#include<stdint.h>
#include<limits.h>
void s8RowQuant(float *A,int m,int n,int ld,float *scalar,MKL_INT8 *a);
void u8ColQuant(float *A,int m,int n,int ld,float *scalar,MKL_INT8 *a);
void s8vecgen(int8_t *a,int n){
	int i;
	for(i=0;i<n;i++){
		a[i] = rand()%256-128;
	}
}
void u8vecgen(uint8_t *a,int n){
	int i;
	for(i=0;i<n;i++){
		a[i] = rand()%256;
	}
}
void s16vecgen(int16_t *a,int n){
	int i;
	for(i=0;i<n;i++){
		a[i] = rand()%(32767+32766)-32767;
	}
}
void s32vecgen(int32_t *a,int n){
	int i;
	for(i=0;i<n;i++){
		a[i] = (int32_t)rand();
	}
}
int main(int argc,char *argv[]){
    mkl_set_num_threads(1);
    int m,n,k;
    srand((unsigned)time(NULL));
    k = 512;
    m = 10000;
    n = m;
    if(argc>1){
    	m = atoi(argv[1]);
        n = m;
    }
    int total = 50000/m;    
    total = (total<10)?10:total;
    MKL_INT32 co = 0;
    int i,j;
    void *a,*b,*c;
    int lda = m;
    int ldb = k;
    int ldc = m;
    double t0;
    printf("====================================\n");
    printf("GEMM m=%d,n=%d,k=%d,nTest=%d\n",m,n,k,total);
    

    //sgemm 
    a = malloc(m*k*sizeof(float));
    b = malloc(k*n*sizeof(float));
    c = malloc(m*n*sizeof(float));

    svecgen(a,m*k,rand());
    svecgen(b,k*n,rand());
    svecgen(c,m*n,rand());
    t0 = get_wtime();
    for(int nTest=0;nTest<total;nTest++){
     	cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,m, n , k, -1, a,
                      lda, b, ldb, 1.0, c, ldc);   
    }
    const double ts = get_wtime() - t0;
    double Rmax = 2*(m/1000.0)*(k/1000.0)*(n/1000.0)/ts*total;
    printf("sgemm t= %lf s, Rmax = %lf GFLOPS\n",ts,Rmax);
    free(a);
    free(b);
    free(c);
    
    
    //s8u8s32 gemm
    a = malloc(m*k*sizeof(int8_t));
    b = malloc(k*n*sizeof(uint8_t));
    c = malloc(m*n*sizeof(int32_t));
    s8vecgen(a,m*k);
    u8vecgen(b,k*n);
    s32vecgen(c,m*n);       
    t0 = get_wtime();
    for(int nTest=0;nTest<total;nTest++){
   	cblas_gemm_s8u8s32(CblasColMajor,CblasNoTrans,CblasNoTrans,CblasFixOffset,\
   		m,n,k,1,a,m,0,b,k,-127,0,c,m,&co);  
      }
    const double t8 = get_wtime() - t0;  
    printf("s8u8s32 t= %lf s,speedup = %lf \n",t8,ts/t8); 
    free(a);
    free(b);
    free(c);
   
       //s16s16s32 gemm
    a = malloc(m*k*sizeof(int16_t));
    b = malloc(k*n*sizeof(int16_t));
    c = malloc(m*n*sizeof(int32_t));
    s16vecgen(a,m*k);
    s16vecgen(b,k*n);
    s32vecgen(c,m*n); 
    t0 = get_wtime();
    for(int nTest=0;nTest<total;nTest++){
   	cblas_gemm_s16s16s32(CblasColMajor,CblasNoTrans,CblasNoTrans,CblasFixOffset,\
   		m,n,k,1,(void*)a,m,0,(void*)b,k,-127,0,(void*)c,m,&co);  
      }
    const double t16 = get_wtime() - t0;  
    printf("s16s16s32 t= %lf s,speedup = %lf \n",t16,ts/t16); 
    free(a);
    free(b);
    free(c);
    
    //s8gemm
    a = malloc(m*k*sizeof(float));
    b = malloc(k*n*sizeof(float));
    c = malloc(m*n*sizeof(float));

    svecgen(a,m*k,rand());
    svecgen(b,k*n,rand());
    svecgen(c,m*n,rand());
    t0 = get_wtime();
    for(int nTest=0;nTest<total;nTest++){
    	q8gemm(m,n,k,a,lda,b,ldb,c,ldc);
    }
    const double tn = get_wtime() - t0;
    printf("q8gemm t= %lf s,speedup = %lf \n",tn,ts/tn);
    free(a);
    free(b);
    free(c);
    printf("====================================\n");
  /*  
    t0 = get_wtime();
    for(int nTest=0;nTest<total;nTest++){
    	s8gemm_mkl(m,n,k,a,lda,b,ldb,c,ldc);
    }

    double to = get_wtime() - t0;
    printf("s8gemm_old t= %lf s, speedup = %lf\n",to,ts/to);
*/

    return 0;
}
