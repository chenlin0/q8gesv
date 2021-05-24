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
/*
 * Do not use  MPI user-defined data types no matter what.  This routine
 * is used for small contiguous messages.
 */
#ifdef HPL_USE_MPI_DATATYPE
#undef HPL_USE_MPI_DATATYPE
#endif

#ifdef STDC_HEADERS
int HPL_sdrv
(
   _D_TYPE_ *                         SBUF,
   int                              SCOUNT,
   int                              STAG,
   _D_TYPE_ *                         RBUF,
   int                              RCOUNT,
   int                              RTAG,
   int                              PARTNER,
   MPI_Comm                         COMM
)
#else
int HPL_sdrv
( SBUF, SCOUNT, STAG, RBUF, RCOUNT, RTAG, PARTNER, COMM )
   _D_TYPE_ *                         SBUF;
   int                              SCOUNT;
   int                              STAG;
   _D_TYPE_ *                         RBUF;
   int                              RCOUNT;
   int                              RTAG;
   int                              PARTNER;
   MPI_Comm                         COMM;
#endif
{
/* 
 * Purpose
 * =======
 *
 * HPL_sdrv is a simple wrapper around MPI_Sendrecv. Its main purpose is
 * to allow for some experimentation and tuning of this simple function.
 * Messages  of  length  less than  or  equal to zero  are not sent  nor
 * received.  Successful completion  is  indicated by the returned error
 * code HPL_SUCCESS.
 *
 * Arguments
 * =========
 *
 * SBUF    (local input)                 _D_TYPE_ *
 *         On entry, SBUF specifies the starting address of buffer to be
 *         sent.
 *
 * SCOUNT  (local input)                 int
 *         On entry,  SCOUNT  specifies  the number  of _D_TYPE_ precision
 *         entries in SBUF. SCOUNT must be at least zero.
 *
 * STAG    (local input)                 int
 *         On entry,  STAG  specifies the message tag to be used for the
 *         sending communication operation.
 *
 * RBUF    (local output)                _D_TYPE_ *
 *         On entry, RBUF specifies the starting address of buffer to be
 *         received.
 *
 * RCOUNT  (local input)                 int
 *         On entry,  RCOUNT  specifies  the number  of _D_TYPE_ precision
 *         entries in RBUF. RCOUNT must be at least zero.
 *
 * RTAG    (local input)                 int
 *         On entry,  RTAG  specifies the message tag to be used for the
 *         receiving communication operation.
 *
 * PARTNER (local input)                 int
 *         On entry,  PARTNER  specifies  the rank of the  collaborative
 *         process in the communication space defined by COMM.
 *
 * COMM    (local input)                 MPI_Comm
 *         The MPI communicator identifying the communication space.
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
return MPI_Sendrecv((void*)SBUF, SCOUNT, _M_TYPE_, PARTNER, STAG,
(void*)RBUF, RCOUNT, _M_TYPE_, PARTNER, RTAG, COMM, MPI_STATUS_IGNORE);
/*
 * End of HPL_sdrv
 */
}
