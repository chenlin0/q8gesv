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

#ifndef HPL_idamax
#ifdef STDC_HEADERS
int Iter_idamax
(
   const int                        N,
   const _ITER_D_TYPE_ *                   X,
   const int                        INCX
)
#else
int Iter_idamax
( N, X, INCX )
   const int                        N;
   const _ITER_D_TYPE_ *                   X;
   const int                        INCX;
#endif 
{
/* 
 * Purpose
 * =======
 *
 * HPL_idamax returns  the index in an n-vector  x  of the first element
 * having maximum absolute value.
 *
 * Arguments
 * =========
 *
 * N       (local input)                 const int
 *         On entry, N specifies the length of the vector x. N  must  be
 *         at least zero.
 *
 * X       (local input)                 const _D_TYPE_ *
 *         On entry,  X  is an incremented array of dimension  at  least
 *         ( 1 + ( n - 1 ) * abs( INCX ) )  that  contains the vector x.
 *
 * INCX    (local input)                 const int
 *         On entry, INCX specifies the increment for the elements of X.
 *         INCX must not be zero.
 *
 * ---------------------------------------------------------------------
 */ 
#ifdef HPL_CALL_CBLAS
#ifdef HPL_AI_DOUBLE
   return( (int)(cblas_idamax( N, X, INCX )) );
#endif

#ifdef HPL_AI_FLOAT
   return( (int)(cblas_idamax( N, X, INCX )) );
#endif

#ifdef HPL_AI_COMPLEX_FLOAT
   register _ITER_D_TYPE_          x0, x1, x2, x3,
                             x4, x5, x6, x7;
   register _ITER_D_TYPE_          absxi, smax = HPL_rzero;
   const _ITER_D_TYPE_             * StX;
   register int              imax = 0, i = 0, j;
   int                       nu;
   const int                 incX2 = 2 * INCX, incX3 = 3 * INCX,
                             incX4 = 4 * INCX, incX5 = 5 * INCX,
                             incX6 = 6 * INCX, incX7 = 7 * INCX,
                             incX8 = 8 * INCX;

   if( N > 0 )
   {
      if( ( nu = ( N >> 3 ) << 3 ) != 0 )
      {
         StX = X + nu * INCX;
 
         do
         {
            x0 = (*X);     x4 = X[incX4]; x1 = X[INCX ]; x5 = X[incX5];
            x2 = X[incX2]; x6 = X[incX6]; x3 = X[incX3]; x7 = X[incX7];
 
            absxi = Mabs( x0 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x1 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x2 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x3 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x4 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x5 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x6 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x7 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
 
            X    += incX8;
 
         } while( X != StX );
      }
 
      for( j = N - nu; j != 0; j-- )
      {
         x0    = (*X);
         absxi = Mabs( x0 ); 
         if( realPart(absxi) > realPart(smax) ){ 
            imax = i; 
            smax = absxi; 
         }
         i    += 1;
         X    += INCX;
      }
   }
   return( imax );
#endif

#ifdef HPL_AI_COMPLEX_DOUBLE
   register _ITER_D_TYPE_          x0, x1, x2, x3,
                             x4, x5, x6, x7;
   register _ITER_D_TYPE_          absxi, smax = HPL_rzero;
   const _ITER_D_TYPE_             * StX;
   register int              imax = 0, i = 0, j;
   int                       nu;
   const int                 incX2 = 2 * INCX, incX3 = 3 * INCX,
                             incX4 = 4 * INCX, incX5 = 5 * INCX,
                             incX6 = 6 * INCX, incX7 = 7 * INCX,
                             incX8 = 8 * INCX;

   if( N > 0 )
   {
      if( ( nu = ( N >> 3 ) << 3 ) != 0 )
      {
         StX = X + nu * INCX;
 
         do
         {
            x0 = (*X);     x4 = X[incX4]; x1 = X[INCX ]; x5 = X[incX5];
            x2 = X[incX2]; x6 = X[incX6]; x3 = X[incX3]; x7 = X[incX7];
 
            absxi = Mabs( x0 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x1 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x2 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x3 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x4 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x5 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x6 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x7 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
 
            X    += incX8;
 
         } while( X != StX );
      }
 
      for( j = N - nu; j != 0; j-- )
      {
         x0    = (*X);
         absxi = Mabs( x0 ); 
         if( realPart(absxi) > realPart(smax) ){ 
            imax = i; 
            smax = absxi; 
         }
         i    += 1;
         X    += INCX;
      }
   }
   return( imax );
   
#endif
#endif
}

#ifdef STDC_HEADERS
int HPL_idamax
(
   const int                        N,
   const _D_TYPE_ *                   X,
   const int                        INCX
)
#else
int HPL_idamax
( N, X, INCX )
   const int                        N;
   const _D_TYPE_ *                   X;
   const int                        INCX;
#endif 
{
/* 
 * Purpose
 * =======
 *
 * HPL_idamax returns  the index in an n-vector  x  of the first element
 * having maximum absolute value.
 *
 * Arguments
 * =========
 *
 * N       (local input)                 const int
 *         On entry, N specifies the length of the vector x. N  must  be
 *         at least zero.
 *
 * X       (local input)                 const _D_TYPE_ *
 *         On entry,  X  is an incremented array of dimension  at  least
 *         ( 1 + ( n - 1 ) * abs( INCX ) )  that  contains the vector x.
 *
 * INCX    (local input)                 const int
 *         On entry, INCX specifies the increment for the elements of X.
 *         INCX must not be zero.
 *
 * ---------------------------------------------------------------------
 */ 
#ifdef HPL_CALL_CBLAS
#ifdef HPL_AI_DOUBLE
   return( (int)(cblas_idamax( N, X, INCX )) );
#endif

#ifdef HPL_AI_FLOAT
   return( (int)(cblas_isamax( N, X, INCX )) );
#endif
#ifdef HPL_AI_COMPLEX_FLOAT
   register _D_TYPE_          x0, x1, x2, x3,
                             x4, x5, x6, x7;
   register _D_TYPE_           absxi, smax = HPL_rzero;
   const _D_TYPE_              * StX;
   register int              imax = 0, i = 0, j;
   int                       nu;
   const int                 incX2 = 2 * INCX, incX3 = 3 * INCX,
                             incX4 = 4 * INCX, incX5 = 5 * INCX,
                             incX6 = 6 * INCX, incX7 = 7 * INCX,
                             incX8 = 8 * INCX;

   if( N > 0 )
   {
      if( ( nu = ( N >> 3 ) << 3 ) != 0 )
      {
         StX = X + nu * INCX;
 
         do
         {
            x0 = (*X);     x4 = X[incX4]; x1 = X[INCX ]; x5 = X[incX5];
            x2 = X[incX2]; x6 = X[incX6]; x3 = X[incX3]; x7 = X[incX7];
 
            absxi = Mabs( x0 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x1 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x2 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x3 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x4 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x5 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x6 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x7 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
 
            X    += incX8;
 
         } while( X != StX );
      }
 
      for( j = N - nu; j != 0; j-- )
      {
         x0    = (*X);
         absxi = Mabs( x0 ); 
         if( realPart(absxi) > realPart(smax) ){ 
            imax = i; 
            smax = absxi; 
         }
         i    += 1;
         X    += INCX;
      }
   }
   return( imax );
#endif
#ifdef HPL_AI_COMPLEX_DOUBLE
   register _D_TYPE_          x0, x1, x2, x3,
                             x4, x5, x6, x7;
   register _D_TYPE_           absxi, smax = HPL_rzero;
   const _D_TYPE_              * StX;
   register int              imax = 0, i = 0, j;
   int                       nu;
   const int                 incX2 = 2 * INCX, incX3 = 3 * INCX,
                             incX4 = 4 * INCX, incX5 = 5 * INCX,
                             incX6 = 6 * INCX, incX7 = 7 * INCX,
                             incX8 = 8 * INCX;

   if( N > 0 )
   {
      if( ( nu = ( N >> 3 ) << 3 ) != 0 )
      {
         StX = X + nu * INCX;
 
         do
         {
            x0 = (*X);     x4 = X[incX4]; x1 = X[INCX ]; x5 = X[incX5];
            x2 = X[incX2]; x6 = X[incX6]; x3 = X[incX3]; x7 = X[incX7];
 
            absxi = Mabs( x0 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x1 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x2 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x3 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x4 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x5 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x6 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
            absxi = Mabs( x7 ); if( realPart(absxi) > realPart(smax) ) { imax = i; smax = absxi; }
            i    += 1;
 
            X    += incX8;
 
         } while( X != StX );
      }
 
      for( j = N - nu; j != 0; j-- )
      {
         x0    = (*X);
         absxi = Mabs( x0 ); 
         if( realPart(absxi) > realPart(smax) ){ 
            imax = i; 
            smax = absxi; 
         }
         i    += 1;
         X    += INCX;
      }
   }
   return( imax );
#endif

#endif


/*
 * End of HPL_idamax
 */
}
 
#endif
