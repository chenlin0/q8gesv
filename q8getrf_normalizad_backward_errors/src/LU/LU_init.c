#include"LU.h"
/*
* init lu
* 分配空间
*/
int LU_init(HPL_LU *lu){
   _D_TYPE_                     * Bptr;
   lu->vptr = NULL;
   static int                 first=1;
   int                        ii, ip2, mycol, myrow, npcol, nprow, nq;
   char                       ctop, cpfact, crfact;
   time_t                     current_time_start, current_time_end;
   HPL_grid_init( MPI_COMM_WORLD, 1,lu->P,lu->Q,
                            &(lu->grid));
   (void) HPL_grid_info(&(lu->grid), &nprow, &npcol, &myrow, &mycol );

   lu->mat.n  = lu->N; 
   lu->mat.nb = lu->NB; 
   lu->mat.info = 0;
   lu->mat.mp = HPL_numroc( lu->N, lu->NB, lu->NB, myrow, 0, nprow );
   nq     = HPL_numroc( lu->N, lu->NB, lu->NB, mycol, 0, npcol );
   lu->mat.nq = nq + 1;
   lu->algo.align = sizeof(_D_TYPE_);
/*
 * Allocate matrix, right-hand-side, and vector solution x. [ A | b ] is
 * N by N+1.  One column is added in every process column for the solve.
 * The  result  however  is stored in a 1 x N vector replicated in every
 * process row. In every process, A is lda * (nq+1), x is 1 * nq and the
 * workspace is mp. 
 *
 * Ensure that lda is a multiple of ALIGN and not a power of 2
 */
   lu->mat.ld = ( ( Mmax( 1, lu->mat.mp ) - 1 ) / lu->algo.align ) * (lu->algo.align);
   do
   {
      ii = ( lu->mat.ld += lu->algo.align ); ip2 = 1;
      while( ii > 1 ) { ii >>= 1; ip2 <<= 1; }
   }
   while( lu->mat.ld == ip2 );
/*
 * Allocate dynamic memory
 */
   lu->vptr = (void*)malloc( ( (size_t)(lu->algo.align) + 
                           (size_t)(lu->mat.ld+1) * (size_t)(lu->mat.nq) ) *
                         sizeof(_D_TYPE_) );


/*
 * generate matrix and right-hand-side, [ A | b ] which is N by N+1.
 */
   lu->mat.A  = (_D_TYPE_ *)HPL_PTR( lu->vptr,
                               ((size_t)(lu->algo.align) * sizeof(_D_TYPE_) ) );
   lu->mat.X  = Mptr( lu->mat.A, 0, lu->mat.nq, lu->mat.ld );
   lu->gflops = NULL;
   return 0;
}