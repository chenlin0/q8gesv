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
#include "hpl.h"
#include<time.h>
#ifdef STDC_HEADERS
void HPL_pdtest
(
   HPL_T_test *                     TEST,
   HPL_T_grid *                     GRID,
   HPL_T_palg *                     ALGO,
   const int                        N,
   const int                        NB
)
#else
void HPL_pdtest
( TEST, GRID, ALGO, N, NB )
   HPL_T_test *                     TEST;
   HPL_T_grid *                     GRID;
   HPL_T_palg *                     ALGO;
   const int                        N;
   const int                        NB;
#endif
{
/* 
 * Purpose
 * =======
 *
 * HPL_pdtest performs  one  test  given a set of parameters such as the
 * process grid, the  problem size, the distribution blocking factor ...
 * This function generates  the data, calls  and times the linear system
 * solver,  checks  the  accuracy  of the  obtained vector solution  and
 * writes this information to the file pointed to by TEST->outfp.
 *
 * Arguments
 * =========
 *
 * TEST    (global input)                HPL_T_test *
 *         On entry,  TEST  points  to a testing data structure:  outfp
 *         specifies the output file where the results will be printed.
 *         It is only defined and used by the process  0  of the  grid.
 *         thrsh  specifies  the  threshhold value  for the test ratio.
 *         Concretely, a test is declared "PASSED"  if and only if the
 *         following inequality is satisfied:
 *         ||Ax-b||_oo / ( epsil *
 *                         ( || x ||_oo * || A ||_oo + || b ||_oo ) *
 *                          N )  < thrsh.
 *         epsil  is the  relative machine precision of the distributed
 *         computer. Finally the test counters, kfail, kpass, kskip and
 *         ktest are updated as follows:  if the test passes,  kpass is
 *         incremented by one;  if the test fails, kfail is incremented
 *         by one; if the test is skipped, kskip is incremented by one.
 *         ktest is left unchanged.
 *
 * GRID    (local input)                 HPL_T_grid *
 *         On entry,  GRID  points  to the data structure containing the
 *         process grid information.
 *
 * ALGO    (global input)                HPL_T_palg *
 *         On entry,  ALGO  points to  the data structure containing the
 *         algorithmic parameters to be used for this test.
 *
 * N       (global input)                const int
 *         On entry,  N specifies the order of the coefficient matrix A.
 *         N must be at least zero.
 *
 * NB      (global input)                const int
 *         On entry,  NB specifies the blocking factor used to partition
 *         and distribute the matrix A. NB must be larger than one.
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
#ifdef HPL_DETAILED_TIMING
   double                     HPL_w[HPL_TIMING_N];
#endif
   double                     wtime[1];
   int                        info[3];
   _REAL_TYPE_             Anorm1, AnormI, Gflops, Xnorm1, XnormI,
                              BnormI, resid0, resid1,temp;
   _ITER_D_TYPE_              * Bptr;
   void                       * vptr = NULL;
   void                       * svptr = NULL;
   void                       * LUvptr = NULL;
   static int                 first=1;
   int                        ii, ip2, mycol, myrow, npcol, nprow, nq;
   char                       ctop, cpfact, crfact;
   time_t                     current_time_start, current_time_end;
   int                        max_it = 50;
   (void) HPL_grid_info( GRID, &nprow, &npcol, &myrow, &mycol );
/* ..
 * .. Executable Statements ..
 */
#ifdef HPL_AI_FLOAT
   HPL_Iter_pmat                mat;
   HPL_T_pmat                 smat;
#endif
#ifdef HPL_AI_DOUBLE
   HPL_T_pmat                 mat;
#endif
#ifdef HPL_AI_COMPLEX_FLOAT
   HPL_Iter_pmat                mat;
   HPL_T_pmat                 smat;
#endif
#ifdef HPL_AI_COMPLEX_DOUBLE
   HPL_T_pmat                 mat;
#endif

   mat.n  = N; mat.nb = NB; mat.info = 0;
   mat.mp = HPL_numroc( N, NB, NB, myrow, 0, nprow );
   nq     = HPL_numroc( N, NB, NB, mycol, 0, npcol );
   mat.nq = nq + 1;
/*
 * Allocate matrix, right-hand-side, and vector solution x. [ A | b ] is
 * N by N+1.  One column is added in every process column for the solve.
 * The  result  however  is stored in a 1 x N vector replicated in every
 * process row. In every process, A is lda * (nq+1), x is 1 * nq and the
 * workspace is mp. 
 *
 * Ensure that lda is a multiple of ALIGN and not a power of 2
 */
   mat.ld = ( ( Mmax( 1, mat.mp ) - 1 ) / ALGO->align ) * ALGO->align;
   do
   {
      ii = ( mat.ld += ALGO->align ); ip2 = 1;
      while( ii > 1 ) { ii >>= 1; ip2 <<= 1; }
   }
   while( mat.ld == ip2 );
/*
 * Allocate dynamic memory
 */
   vptr = (void*)malloc( ( (size_t)(ALGO->align) + 
                           (size_t)(mat.ld+1) * (size_t)(mat.nq) ) *
                         sizeof(_ITER_D_TYPE_) );
   info[0] = (vptr == NULL); 
   info[1] = myrow; 
   info[2] = mycol;
   (void) HPL_all_reduce( (void *)(info), 3, HPL_INT, HPL_max,
                          GRID->all_comm );

   if( info[0] != 0 )
   {
      if( ( myrow == 0 ) && ( mycol == 0 ) )
         HPL_pwarn( TEST->outfp, __LINE__, "HPL_pdtest",
                    "[%d,%d] %s", info[1], info[2],
                    "Memory allocation failed for A, x and b. Skip." );
      (TEST->kskip)++;
      /* some processes might have succeeded with allocation */
      if (vptr) free(vptr);
      return;
   }
/*
 * generate matrix and right-hand-side, [ A | b ] which is N by N+1.
 */

   mat.A  = (_ITER_D_TYPE_ *)HPL_PTR( vptr,
                               ((size_t)(ALGO->align) * sizeof(_ITER_D_TYPE_) ) );
   mat.X  = Mptr( mat.A, 0, mat.nq, mat.ld );
   int seed=time(NULL);
   HPL_pdmatgen( GRID, N, N+1, NB, mat.A, mat.ld, seed );
   Bptr = Mptr( mat.A, 0, nq, mat.ld );
   #ifdef HPL_AI_FLOAT

   smat.n  = mat.n; 
   smat.nb = mat.nb; 
   smat.info = 0;
   smat.mp = mat.mp;
   smat.nq = mat.nq;
   smat.ld = mat.ld;
   svptr = (void*)malloc( ( (size_t)(ALGO->align) + 
                           (size_t)(smat.ld+1) * (size_t)(smat.nq) ) *
                         sizeof(_D_TYPE_) );
   smat.A  = (_D_TYPE_ *)HPL_PTR( svptr,
                               ((size_t)(ALGO->align) * sizeof(_D_TYPE_) ) );
   smat.X  = Mptr( smat.A, 0, smat.nq, smat.ld );

   #endif

/*
 * Solve linear system
 */
   HPL_ptimer_boot(); (void) HPL_barrier( GRID->all_comm );
   time( &current_time_start );
   HPL_ptimer( 0 );
   #ifdef HPL_AI_FLOAT

   convert_double_to_float(mat.A, mat.ld, smat.A,
                             smat.ld, smat.mp, smat.nq);
   convert_double_to_float(mat.X, 1, smat.X,
                             1,smat.mp, 1);
   HPL_pdgesv( GRID, ALGO, &smat );
   convert_float_to_double(smat.X, 1, mat.X,1,mat.mp, 1);
   convert_float_to_double(smat.A, smat.ld, mat.A,mat.ld, mat.mp, mat.nq);
   if( svptr ) free( svptr );

   
   #endif
   #ifdef HPL_AI_DOUBLE
      HPL_pdgesv( GRID, ALGO, &mat );
   #endif
   #ifdef HPL_AI_COMPLEX_FLOAT
   convert_double_to_float(mat.A, mat.ld, smat.A,
                             smat.ld, smat.mp, smat.nq);
   convert_double_to_float(mat.X, 1, smat.X,
                             1,smat.mp, 1);
   HPL_pdgesv( GRID, ALGO, &smat );
   convert_float_to_double(smat.X, 1, mat.X,1,mat.mp, 1);
   convert_float_to_double(smat.A, smat.ld, LU,smat.ld, smat.mp, smat.nq);
   if( svptr ) free( svptr );

   #endif
   #ifdef HPL_AI_COMPLEX_DOUBLE
      HPL_pdgesv( GRID, ALGO, &mat );
   #endif
   HPL_ptimer( 0 );
   time( &current_time_end );


/*
 * Gather max of all CPU and WALL clock timings and print timing results
 */
   HPL_ptimer_combine( GRID->all_comm, HPL_AMAX_PTIME, HPL_WALL_PTIME,
                       1, 0, wtime );

   if( ( myrow == 0 ) && ( mycol == 0 ) )
   {
      if( first )
      {
         HPL_fprintf( TEST->outfp, "%s%s\n",
                      "========================================",
                      "========================================" );
         HPL_fprintf( TEST->outfp, "%s%s\n",
                      "T/V                N    NB     P     Q",
                      "               Time                 Gflops" );
         HPL_fprintf( TEST->outfp, "%s%s\n",
                      "----------------------------------------",
                      "----------------------------------------" );
         if( realPart(TEST->thrsh) <= HPL_rzero ) first = 0;
      }
/*
 * 2/3 N^3 - 1/2 N^2 flops for LU factorization + 2 N^2 flops for solve.
 * Print WALL time
 */
#ifdef HPL_AI_FLOAT
      Gflops = ( ( (double)(N) /   1.0e+9 ) * 
                 ( (double)(N) / wtime[0] ) ) * 
                 ( ( 2.0 / 3.0 ) * (double)(N) + ( 3.0 / 2.0 ) );
#endif
#ifdef HPL_AI_DOUBLE
      Gflops = ( ( (double)(N) /   1.0e+9 ) * 
                 ( (double)(N) / wtime[0] ) ) * 
                 ( ( 2.0 / 3.0 ) * (double)(N) + ( 3.0 / 2.0 ) );
#endif
#ifdef HPL_AI_COMPLEX_FLOAT
      Gflops = ( ( (double)(N) /   1.0e+9 ) * 
                 ( (double)(N) / wtime[0] ) ) * 
                 ( ( 2.0 ) * (double)(N) + ( 19.0 / 2.0 ) );
#endif
#ifdef HPL_AI_COMPLEX_DOUBLE
      Gflops = ( ( (double)(N) /   1.0e+9 ) * 
                 ( (double)(N) / wtime[0] ) ) * 
                 ( ( 2.0 ) * (double)(N) );
#endif
      cpfact = ( ( (HPL_T_FACT)(ALGO->pfact) == 
                   (HPL_T_FACT)(HPL_LEFT_LOOKING) ) ?  (char)('L') :
                 ( ( (HPL_T_FACT)(ALGO->pfact) == (HPL_T_FACT)(HPL_CROUT) ) ?
                   (char)('C') : (char)('R') ) );
      crfact = ( ( (HPL_T_FACT)(ALGO->rfact) == 
                   (HPL_T_FACT)(HPL_LEFT_LOOKING) ) ?  (char)('L') :
                 ( ( (HPL_T_FACT)(ALGO->rfact) == (HPL_T_FACT)(HPL_CROUT) ) ? 
                   (char)('C') : (char)('R') ) );

      if(      ALGO->btopo == HPL_1RING   ) ctop = '0';
      else if( ALGO->btopo == HPL_1RING_M ) ctop = '1';
      else if( ALGO->btopo == HPL_2RING   ) ctop = '2';
      else if( ALGO->btopo == HPL_2RING_M ) ctop = '3';
      else if( ALGO->btopo == HPL_BLONG   ) ctop = '4';
      else /* if( ALGO->btopo == HPL_BLONG_M ) */ ctop = '5';

      if( wtime[0] > HPL_rzero ) {
         HPL_fprintf( TEST->outfp,
             "W%c%1d%c%c%1d%c%1d%12d %5d %5d %5d %18.2f    %19.4e\n",
             ( GRID->order == HPL_ROW_MAJOR ? 'R' : 'C' ),
             ALGO->depth, ctop, crfact, ALGO->nbdiv, cpfact, ALGO->nbmin,
             N, NB, nprow, npcol, wtime[0], Gflops );
         HPL_fprintf( TEST->outfp,
             "HPL_pdgesv() start time %s\n", ctime( &current_time_start ) );
         HPL_fprintf( TEST->outfp,
             "HPL_pdgesv() end time   %s\n", ctime( &current_time_end ) );
      }
   }
#ifdef HPL_DETAILED_TIMING
   HPL_ptimer_combine( GRID->all_comm, HPL_AMAX_PTIME, HPL_WALL_PTIME,
                       HPL_TIMING_N, HPL_TIMING_BEG, HPL_w );
   if( ( myrow == 0 ) && ( mycol == 0 ) )
   {
      HPL_fprintf( TEST->outfp, "%s%s\n",
                   "--VVV--VVV--VVV--VVV--VVV--VVV--VVV--V",
                   "VV--VVV--VVV--VVV--VVV--VVV--VVV--VVV-" );
/*
 * Recursive panel factorization
 */
      if( HPL_w[HPL_TIMING_RPFACT-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Max aggregated wall time rfact . . . : %18.2f\n",
                      HPL_w[HPL_TIMING_RPFACT-HPL_TIMING_BEG] );
/*
 * Panel factorization
 */
      if( HPL_w[HPL_TIMING_PFACT-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "+ Max aggregated wall time pfact . . : %18.2f\n",
                      HPL_w[HPL_TIMING_PFACT-HPL_TIMING_BEG] );
/*
 * Panel factorization (swap)
 */
      if( HPL_w[HPL_TIMING_MXSWP-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "+ Max aggregated wall time mxswp . . : %18.2f\n",
                      HPL_w[HPL_TIMING_MXSWP-HPL_TIMING_BEG] );
/*
 * Update
 */
      if( HPL_w[HPL_TIMING_UPDATE-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Max aggregated wall time update  . . : %18.2f\n",
                      HPL_w[HPL_TIMING_UPDATE-HPL_TIMING_BEG] );
/*
 * Update (swap)
 */
      if( HPL_w[HPL_TIMING_LASWP-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "+ Max aggregated wall time laswp . . : %18.2f\n",
                      HPL_w[HPL_TIMING_LASWP-HPL_TIMING_BEG] );
/*
 * Upper triangular system solve
 */
      if( HPL_w[HPL_TIMING_PTRSV-HPL_TIMING_BEG] > HPL_rzero )
         HPL_fprintf( TEST->outfp,
                      "Max aggregated wall time up tr sv  . : %18.2f\n",
                      HPL_w[HPL_TIMING_PTRSV-HPL_TIMING_BEG] );

      if( realPart(TEST->thrsh) <= HPL_rzero )
         HPL_fprintf( TEST->outfp, "%s%s\n",
                      "========================================",
                      "========================================" );
   }
#endif
/*
 * Quick return, if I am not interested in checking the computations
 */
   if( realPart(TEST->thrsh) <= HPL_rzero )
   { (TEST->kpass)++; if( vptr ) free( vptr ); return; }
/*
 * Check info returned by solve
 */


   if( mat.info != 0 )
   {
      if( ( myrow == 0 ) && ( mycol == 0 ) )
         HPL_pwarn( TEST->outfp, __LINE__, "HPL_pdtest", "%s %d, %s", 
                    "Error code returned by solve is", mat.info, "skip" );
      (TEST->kskip)++;
      if( vptr ){
         free( vptr );
      } 
      return;
   }


/*
 * Check computation, re-generate [ A | b ], compute norm 1 and inf of A and x,
 * and norm inf of b - A x. Display residual checks.
 */
   HPL_pdmatgen( GRID, N, N+1, NB, mat.A, mat.ld, HPL_ISEED );
   Anorm1 = HPL_pdlange( GRID, HPL_NORM_1, N, N, NB, mat.A, mat.ld );
   AnormI = HPL_pdlange( GRID, HPL_NORM_I, N, N, NB, mat.A, mat.ld );
/*
 * Because x is distributed in process rows, switch the norms
 */
   XnormI = HPL_pdlange( GRID, HPL_NORM_1, 1, N, NB, mat.X, 1 );
   Xnorm1 = HPL_pdlange( GRID, HPL_NORM_I, 1, N, NB, mat.X, 1 );
   


/*
 * If I am in the col that owns b, (1) compute local BnormI, (2) all_reduce to
 * find the max (in the col). Then (3) broadcast along the rows so that every
 * process has BnormI. Note that since we use a uniform distribution in [-0.5,0.5]
 * for the entries of B, it is very likely that BnormI (<=,~) 0.5.
 */
   Bptr = Mptr( mat.A, 0, nq, mat.ld );
   if( mycol == HPL_indxg2p( N, NB, NB, 0, npcol ) ){
      if( mat.mp > 0 )
      {
         BnormI = Mabs(Bptr[Iter_idamax( mat.mp, Bptr, 1 )]);
      }
      else
      {
         BnormI = HPL_rzero;
      }
      MPI_Allreduce((void*)(&BnormI),(void*)(&temp),1,MPI_DOUBLE,MPI_MAX,GRID->col_comm);

   }
   BnormI = temp;
   MPI_Bcast((void *)(&BnormI), 1,MPI_DOUBLE,HPL_indxg2p( N, NB, NB, 0, npcol ),
                          GRID->row_comm);
/*
 * If I own b, compute ( b - A x ) and ( - A x ) otherwise
 */
   if( mycol == HPL_indxg2p( N, NB, NB, 0, npcol ) )
   {
      Iter_dgemv( HplColumnMajor, HplNoTrans, mat.mp, nq, -HPL_rone,
                 mat.A, mat.ld, mat.X, 1, HPL_rone, Bptr, 1 );
   }
   else if( nq > 0 )
   {
      Iter_dgemv( HplColumnMajor, HplNoTrans, mat.mp, nq, -HPL_rone,
                 mat.A, mat.ld, mat.X, 1, HPL_rzero, Bptr, 1 );
   }
   else { for( ii = 0; ii < mat.mp; ii++ ) Bptr[ii] = HPL_rzero; }
/*
 * Reduce the distributed residual in process column 0
 */
   if( mat.mp > 0 )
      (void) Iter_reduce( Bptr, mat.mp, HPL_DOUBLE, HPL_sum, 0,
                         GRID->row_comm );
/*
 * Compute || b - A x ||_oo
 */
   resid0 = HPL_pdlange( GRID, HPL_NORM_I, N, 1, NB, Bptr, mat.ld );
   double res = resid0/(AnormI * XnormI);
   printf("N=%d,||Ax-b||_oo/(||A||_oo*||x||_oo)=%16.8e\n",N,res);
/*
 * Computes and displays norms, residuals ...
 */
/*
   if( N <= 0 )
   {
      resid1 = HPL_rzero;
   }
   else
   {
      resid1 = resid0 / ( realPart(TEST->epsil) * ( AnormI * XnormI + BnormI ) * (double)(N) );
   }
  
   if( realPart(resid1) < realPart(TEST->thrsh) ) (TEST->kpass)++;
   else                       (TEST->kfail)++;

   if( ( myrow == 0 ) && ( mycol == 0 ) )
   {
      HPL_fprintf( TEST->outfp, "%s%s\n",
                   "----------------------------------------",
                   "----------------------------------------" );
      HPL_fprintf( TEST->outfp, "%s%16.8e%s%s\n",
         "||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)= ", realPart(resid1),
         " ...... ", ( realPart(resid1) < realPart(TEST->thrsh) ? "PASSED" : "FAILED" ) );

      if( realPart(resid1) > realPart(TEST->thrsh)){
         HPL_fprintf( TEST->outfp, "%s%18.6f\n",
         "||Ax-b||_oo  . . . . . . . . . . . . . . . . . = ", realPart(resid0) );
         HPL_fprintf( TEST->outfp, "%s%18.6f\n",
         "||A||_oo . . . . . . . . . . . . . . . . . . . = ", realPart(AnormI) );
         HPL_fprintf( TEST->outfp, "%s%18.6f\n",
         "||A||_1  . . . . . . . . . . . . . . . . . . . = ", realPart(Anorm1) );
         HPL_fprintf( TEST->outfp, "%s%18.6f\n",
         "||x||_oo . . . . . . . . . . . . . . . . . . . = ", realPart(XnormI) );
         HPL_fprintf( TEST->outfp, "%s%18.6f\n",
         "||x||_1  . . . . . . . . . . . . . . . . . . . = ", realPart(Xnorm1) );
         HPL_fprintf( TEST->outfp, "%s%18.6f\n",
         "||b||_oo . . . . . . . . . . . . . . . . . . . = ", realPart(BnormI) );
      }
   }
   */
   if( vptr ) free( vptr );
/*
 * End of HPL_pdtest
 */
}
