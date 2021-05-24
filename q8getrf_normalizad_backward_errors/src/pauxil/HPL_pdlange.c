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

#ifdef STDC_HEADERS
_REAL_TYPE_ HPL_pdlange
(
   const HPL_T_grid *               GRID,
   const HPL_T_NORM                 NORM,
   const int                        M,
   const int                        N,
   const int                        NB,
   const _ITER_D_TYPE_ *                   A,
   const int                        LDA
)
#else
_REAL_TYPE_ HPL_pdlange
( GRID, NORM, M, N, NB, A, LDA )
   const HPL_T_grid *               GRID;
   const HPL_T_NORM                 NORM;
   const int                        M;
   const int                        N;
   const int                        NB;
   const _ITER_D_TYPE_ *                   A;
   const int                        LDA;
#endif
{
/* 
 * Purpose
 * =======
 *
 * HPL_pdlange returns  the value of the one norm,  or the infinity norm,
 * or the element of largest absolute value of a distributed matrix A:  
 *  
 *  
 *    max(abs(A(i,j))) when NORM = HPL_NORM_A,                          
 *    norm1(A),        when NORM = HPL_NORM_1,                          
 *    normI(A),        when NORM = HPL_NORM_I,                          
 *  
 * where norm1 denotes the one norm of a matrix (maximum column sum) and
 * normI denotes  the infinity norm of a matrix (maximum row sum).  Note
 * that max(abs(A(i,j))) is not a matrix norm.
 *
 * Arguments
 * =========
 *
 * GRID    (local input)                 const HPL_T_grid *
 *         On entry,  GRID  points  to the data structure containing the
 *         process grid information.
 *
 * NORM    (global input)                const HPL_T_NORM
 *         On entry,  NORM  specifies  the  value to be returned by this
 *         function as described above.
 *
 * M       (global input)                const int
 *         On entry,  M  specifies  the number  of rows of the matrix A.
 *         M must be at least zero.
 *
 * N       (global input)                const int
 *         On entry,  N specifies the number of columns of the matrix A.
 *         N must be at least zero.
 *
 * NB      (global input)                const int
 *         On entry,  NB specifies the blocking factor used to partition
 *         and distribute the matrix. NB must be larger than one.
 *
 * A       (local input)                 const _D_TYPE_ *
 *         On entry,  A  points to an array of dimension  (LDA,LocQ(N)),
 *         that contains the local pieces of the distributed matrix A.
 *
 * LDA     (local input)                 const int
 *         On entry, LDA specifies the leading dimension of the array A.
 *         LDA must be at least max(1,LocP(M)).
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
   _REAL_TYPE_                s, v0=HPL_rzero, * work = NULL,temp,* workTemp = NULL;
   MPI_Comm                   Acomm, Ccomm, Rcomm;
   int                        ii, jj, mp, mycol, myrow, npcol, nprow,
                              nq;
/* ..
 * .. Executable Statements ..
 */
   (void) HPL_grid_info( GRID, &nprow, &npcol, &myrow, &mycol );
   Rcomm = GRID->row_comm; Ccomm = GRID->col_comm;
   Acomm = GRID->all_comm;

   Mnumroc( mp, M, NB, NB, myrow, 0, nprow );
   Mnumroc( nq, N, NB, NB, mycol, 0, npcol );

   if( Mmin( M, N ) == 0 ) { return( v0 ); }
   else if( NORM == HPL_NORM_A )
   {
/*
 * max( abs( A ) )
 */
      if( ( nq > 0 ) && ( mp > 0 ) )
      {
         for( jj = 0; jj < nq; jj++ )
         {
            for( ii = 0; ii < mp; ii++ )
            { v0 = Mmax( v0, Mabs( *A ) ); A++; }
            A += LDA - mp;
         }
      }
      MPI_Reduce((void *)(&v0), (void*)(&temp),1, MPI_DOUBLE, MPI_MAX, 0 , Acomm);
      v0 = temp;
   }
   else if( NORM == HPL_NORM_1 )
   {
/*
 * Find norm_1( A ).
 */
      if( nq > 0 )
      {
         workTemp = (_REAL_TYPE_*)malloc( (size_t)(nq) * sizeof( _REAL_TYPE_) );
         work = (_REAL_TYPE_*)malloc( (size_t)(nq) * sizeof( _REAL_TYPE_) );
         if( (workTemp == NULL)||(work == NULL) )
         { HPL_pabort( __LINE__, "HPL_pdlange", "Memory allocation failed" ); }

         for( jj = 0; jj < nq; jj++ )
         {
            s = HPL_rzero;
            for( ii = 0; ii < mp; ii++ ) { s += Mabs( *A ); A++; }
            workTemp[jj] = s; A += LDA - mp;
         }
/*
 * Find sum of global matrix columns, store on row 0 of process grid
 */
         MPI_Reduce((void*)workTemp,(void*)work,nq,MPI_DOUBLE,MPI_SUM,0,Ccomm);
         if( workTemp ){
             free( workTemp );
         }
/*
 * Find maximum sum of columns for 1-norm
 */
         if( myrow == 0 )
         { v0 = work[cblas_idamax( nq, work, 1 )]; v0 = Mabs( v0 ); }
         if( work ){
             free( work );
         }
      }
/*
 * Find max in row 0, store result in process (0,0)
 */
      if( myrow == 0 ){
         MPI_Reduce((void *)(&v0),(void *)(&temp),1,MPI_DOUBLE,MPI_MAX,0,Rcomm); 
      }
      v0 = temp;
   }
   else if( NORM == HPL_NORM_I )
   {
/*
 * Find norm_inf( A )
 */
      if( mp > 0 )
      {
         workTemp = (_REAL_TYPE_*)malloc( (size_t)(mp) * sizeof( _REAL_TYPE_) );
         work = (_REAL_TYPE_*)malloc( (size_t)(mp) * sizeof( _REAL_TYPE_) );
         if( (workTemp == NULL)||(work == NULL) )
         { HPL_pabort( __LINE__, "HPL_pdlange", "Memory allocation failed" ); }

         for( ii = 0; ii < mp; ii++ ) { workTemp[ii] = HPL_rzero; }

         for( jj = 0; jj < nq; jj++ )
         {
            for( ii = 0; ii < mp; ii++ )
            { workTemp[ii] += Mabs( *A ); A++; }
            A += LDA - mp;
         }
/*       
 * Find sum of global matrix rows, store on column 0 of process grid
 */      
         MPI_Reduce((void*)workTemp,(void*)work,mp,MPI_DOUBLE,MPI_SUM,0,Rcomm);
         if( workTemp ){
             free( workTemp );
         }

/*       
 * Find maximum sum of rows for inf-norm
 */      
         if( mycol == 0 )
         { v0 = work[cblas_idamax( mp, work, 1 )]; v0 = Mabs( v0 ); }
         if( work ) free( work );
      }
/*
 * Find max in column 0, store result in process (0,0)
 */
      if( mycol == 0 )
      {
         MPI_Reduce((void*)(&v0),(void*)(&temp),1,MPI_DOUBLE,MPI_MAX,0,Ccomm);
      }
      
   }
/*
 * Broadcast answer to every process in the grid
 */
   MPI_Bcast((void*)(&v0),1,MPI_DOUBLE,0,Acomm);
   return( v0 );
/*
 * End of HPL_pdlange
 */
}
