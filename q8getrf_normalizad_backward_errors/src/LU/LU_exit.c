#include"LU.h"
int LU_exit(HPL_LU *lu){
   if(lu->vptr){
      free(lu->vptr);
   }
   if(lu->gflops){
      free(lu->gflops);
   }  
   (void) HPL_grid_exit( &(lu->grid) );
   return 0;
}