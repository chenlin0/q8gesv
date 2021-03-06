#include"LU.h"
void LU_pdinfo(
   HPL_T_test *                     TEST,
   int *                            NS,
   int *                            N,
   int *                            NBS,
   int *                            NB,
   HPL_T_ORDER *                    PMAPPIN,
   int *                            NPQS,
   int *                            P,
   int *                            Q,
   int *                            NPFS,
   HPL_T_FACT *                     PF,
   int *                            NBMS,
   int *                            NBM,
   int *                            NDVS,
   int *                            NDV,
   int *                            NRFS,
   HPL_T_FACT *                     RF,
   int *                            NTPS,
   HPL_T_TOP *                      TP,
   int *                            NDHS,
   int *                            DH,
   HPL_T_SWAP *                     FSWAP,
   int *                            TSWAP,
   int *                            L1NOTRAN,
   int *                            UNOTRAN,
   int *                            EQUIL,
   int *                            ALIGN
){
/* 
 * Purpose
 * =======
 *
 * LU_pdinfo reads  the  startup  information for the various tests and
 * transmits it to all processes.
 *
 * Arguments
 * =========
 *
 * TEST    (global output)               HPL_T_test *
 *         On entry, TEST  points to a testing data structure.  On exit,
 *         the fields of this data structure are initialized as follows:
 *         TEST->outfp  specifies the output file where the results will
 *         be printed.  It is only defined and used by  the process 0 of
 *         the grid.  TEST->thrsh specifies the threshhold value for the
 *         test ratio.  TEST->epsil is the relative machine precision of
 *         the distributed computer.  Finally  the test counters, kfail,
 *         kpass, kskip, ktest are initialized to zero.
 *
 * NS      (global output)               int *
 *         On exit,  NS  specifies the number of different problem sizes
 *         to be tested. NS is less than or equal to HPL_MAX_PARAM.
 *
 * N       (global output)               int *
 *         On entry, N is an array of dimension HPL_MAX_PARAM.  On exit,
 *         the first NS entries of this array contain the  problem sizes
 *         to run the code with.
 *
 * NBS     (global output)               int *
 *         On exit,  NBS  specifies the number of different distribution
 *         blocking factors to be tested. NBS must be less than or equal
 *         to HPL_MAX_PARAM.
 *
 * NB      (global output)               int *
 *         On exit,  PMAPPIN  specifies the process mapping onto the no-
 *         des of the  MPI machine configuration.  PMAPPIN  defaults  to
 *         row-major ordering.
 *
 * PMAPPIN (global output)               HPL_T_ORDER *
 *         On entry, NB is an array of dimension HPL_MAX_PARAM. On exit,
 *         the first NBS entries of this array contain the values of the
 *         various distribution blocking factors, to run the code with.
 *
 * NPQS    (global output)               int *
 *         On exit, NPQS  specifies the  number of different values that
 *         can be used for P and Q, i.e., the number of process grids to
 *         run  the  code with.  NPQS must be  less  than  or  equal  to
 *         HPL_MAX_PARAM.
 *
 * P       (global output)               int *
 *         On entry, P  is an array of dimension HPL_MAX_PARAM. On exit,
 *         the first NPQS entries of this array contain the values of P,
 *         the number of process rows of the  NPQS grids to run the code
 *         with.
 *
 * Q       (global output)               int *
 *         On entry, Q  is an array of dimension HPL_MAX_PARAM. On exit,
 *         the first NPQS entries of this array contain the values of Q,
 *         the number of process columns of the  NPQS  grids to  run the
 *         code with.
 *
 * NPFS    (global output)               int *
 *         On exit, NPFS  specifies the  number of different values that
 *         can be used for PF : the panel factorization algorithm to run
 *         the code with. NPFS is less than or equal to HPL_MAX_PARAM.
 *
 * PF      (global output)               HPL_T_FACT *
 *         On entry, PF is an array of dimension HPL_MAX_PARAM. On exit,
 *         the first  NPFS  entries  of this array  contain  the various
 *         panel factorization algorithms to run the code with.
 *
 * NBMS    (global output)               int *
 *         On exit,  NBMS  specifies  the  number  of  various recursive
 *         stopping criteria  to be tested.  NBMS  must be  less than or
 *         equal to HPL_MAX_PARAM.
 *
 * NBM     (global output)               int *
 *         On entry,  NBM  is an array of  dimension  HPL_MAX_PARAM.  On
 *         exit, the first NBMS entries of this array contain the values
 *         of the various recursive stopping criteria to be tested.
 *
 * NDVS    (global output)               int *
 *         On exit,  NDVS  specifies  the number  of various numbers  of
 *         panels in recursion to be tested.  NDVS is less than or equal
 *         to HPL_MAX_PARAM.
 *
 * NDV     (global output)               int *
 *         On entry,  NDV  is an array of  dimension  HPL_MAX_PARAM.  On
 *         exit, the first NDVS entries of this array contain the values
 *         of the various numbers of panels in recursion to be tested.
 *
 * NRFS    (global output)               int *
 *         On exit, NRFS  specifies the  number of different values that
 *         can be used for RF : the recursive factorization algorithm to
 *         be tested. NRFS is less than or equal to HPL_MAX_PARAM.
 *
 * RF      (global output)               HPL_T_FACT *
 *         On entry, RF is an array of dimension HPL_MAX_PARAM. On exit,
 *         the first  NRFS  entries  of  this array contain  the various
 *         recursive factorization algorithms to run the code with.
 *
 * NTPS    (global output)               int *
 *         On exit, NTPS  specifies the  number of different values that
 *         can be used for the  broadcast topologies  to be tested. NTPS
 *         is less than or equal to HPL_MAX_PARAM.
 *
 * TP      (global output)               HPL_T_TOP *
 *         On entry, TP is an array of dimension HPL_MAX_PARAM. On exit,
 *         the  first NTPS  entries of this  array  contain  the various
 *         broadcast (along rows) topologies to run the code with.
 *
 * NDHS    (global output)               int *
 *         On exit, NDHS  specifies the  number of different values that
 *         can be used for the  lookahead depths to be  tested.  NDHS is
 *         less than or equal to HPL_MAX_PARAM.
 *
 * DH      (global output)               int *
 *         On entry,  DH  is  an array of  dimension  HPL_MAX_PARAM.  On
 *         exit, the first NDHS entries of this array contain the values
 *         of lookahead depths to run the code with.  Such a value is at
 *         least 0 (no-lookahead) or greater than zero.
 *
 * FSWAP   (global output)               HPL_T_SWAP *
 *         On exit, FSWAP specifies the swapping algorithm to be used in
 *         all tests.
 *
 * TSWAP   (global output)               int *
 *         On exit,  TSWAP  specifies the swapping threshold as a number
 *         of columns when the mixed swapping algorithm was chosen.
 *
 * L1NOTRA (global output)               int *
 *         On exit, L1NOTRAN specifies whether the upper triangle of the
 *         panels of columns  should  be stored  in  no-transposed  form
 *         (L1NOTRAN=1) or in transposed form (L1NOTRAN=0).
 *
 * UNOTRAN (global output)               int *
 *         On exit, UNOTRAN  specifies whether the panels of rows should
 *         be stored in  no-transposed form  (UNOTRAN=1)  or  transposed
 *         form (UNOTRAN=0) during their broadcast.
 *
 * EQUIL   (global output)               int *
 *         On exit,  EQUIL  specifies  whether  equilibration during the
 *         swap-broadcast  of  the  panel of rows  should  be  performed
 *         (EQUIL=1) or not (EQUIL=0).
 *
 * ALIGN   (global output)               int *
 *         On exit,  ALIGN  specifies the alignment  of  the dynamically
 *         allocated buffers in _D_TYPE_ precision words. ALIGN is greater
 *         than zero.
 *
 * ---------------------------------------------------------------------
 */ 
/*
 * .. Local Variables ..
 */
   char                       file[HPL_LINE_MAX], line[HPL_LINE_MAX],
                              auth[HPL_LINE_MAX], num [HPL_LINE_MAX];
   FILE                       * infp;
   int                        * iwork = NULL;
   char                       * lineptr;
   int                        error=0, fid, i, j, lwork, maxp, nprocs,
                              rank, size;
/* ..
 * .. Executable Statements ..
 */
   MPI_Comm_rank( MPI_COMM_WORLD, &rank );
   MPI_Comm_size( MPI_COMM_WORLD, &size );
/*
 * Initialize the TEST data structure with default values
 */
   TEST->outfp = stderr; TEST->epsil = 2.0e-16; TEST->thrsh = 16.0;
   TEST->kfail = TEST->kpass = TEST->kskip = TEST->ktest = 0;
/*
 * Process 0 reads the input data, broadcasts to other processes and
 * writes needed information to TEST->outfp.
 */
   if( rank == 0 )
   {
/*
 * Open file and skip data file header
 */
      if( ( infp = fopen( "LU.dat", "r" ) ) == NULL )
      { 
         HPL_pwarn( stderr, __LINE__, "LU_pdinfo",
                    "cannot open file HPL.dat" );
         error = 1; goto label_error;
      }

      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) fgets( auth, HPL_LINE_MAX - 2, infp );
/*
 * Read name and unit number for summary output file
 */
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", file );
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", num  );
      fid = atoi( num );
      if     ( fid == 6 ) TEST->outfp = stdout;
      else if( fid == 7 ) TEST->outfp = stderr;
      else if( ( TEST->outfp = fopen( file, "w" ) ) == NULL )
      {
         HPL_pwarn( stderr, __LINE__, "LU_pdinfo", "cannot open file %s.",
                    file );
         error = 1; goto label_error;
      }


/*
 * Checking threshold value (TEST->thrsh)
 */
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", num ); TEST->thrsh = atof( num );
/*
 * Panel factorization algorithm (PF)
 */
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", num ); *NPFS = atoi( num );
      if( ( *NPFS < 1 ) || ( *NPFS > HPL_MAX_PARAM ) )
      {
         HPL_pwarn( stderr, __LINE__, "LU_pdinfo", "%s %s %d",
                    "number of values of PFACT",
                    "is less than 1 or greater than", HPL_MAX_PARAM );
         error = 1; goto label_error;
      }
      (void) fgets( line, HPL_LINE_MAX - 2, infp ); lineptr = line;
      for( i = 0; i < *NPFS; i++ )
      {
         (void) sscanf( lineptr, "%s", num ); lineptr += strlen( num ) + 1;
         j = atoi( num );
         if(      j == 0 ) PF[ i ] = HPL_LEFT_LOOKING;
         else if( j == 1 ) PF[ i ] = HPL_CROUT;
         else if( j == 2 ) PF[ i ] = HPL_RIGHT_LOOKING;
         else              PF[ i ] = HPL_RIGHT_LOOKING;
      }
/*
 * Recursive stopping criterium (>=1) (NBM)
 */
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", num ); *NBMS = atoi( num );
      if( ( *NBMS < 1 ) || ( *NBMS > HPL_MAX_PARAM ) )
      {
         HPL_pwarn( stderr, __LINE__, "HPL_pdinfo", "%s %s %d",
                    "Number of values of NBMIN",
                    "is less than 1 or greater than", HPL_MAX_PARAM );
         error = 1; goto label_error;
      }
      (void) fgets( line, HPL_LINE_MAX - 2, infp ); lineptr = line;
      for( i = 0; i < *NBMS; i++ )
      {
         (void) sscanf( lineptr, "%s", num ); lineptr += strlen( num ) + 1;
         if( ( NBM[ i ] = atoi( num ) ) < 1 )
         {
            HPL_pwarn( stderr, __LINE__, "HPL_pdinfo",
                       "Value of NBMIN less than 1" );
            error = 1; goto label_error;
         }
      }
/*
 * Number of panels in recursion (>=2) (NDV)
 */
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", num ); *NDVS = atoi( num );
      if( ( *NDVS < 1 ) || ( *NDVS > HPL_MAX_PARAM ) )
      {
         HPL_pwarn( stderr, __LINE__, "LU_pdinfo", "%s %s %d",
                    "Number of values of NDIV",
                    "is less than 1 or greater than", HPL_MAX_PARAM );
         error = 1; goto label_error;
      }
      (void) fgets( line, HPL_LINE_MAX - 2, infp ); lineptr = line;
      for( i = 0; i < *NDVS; i++ )
      {
         (void) sscanf( lineptr, "%s", num ); lineptr += strlen( num ) + 1;
         if( ( NDV[ i ] = atoi( num ) ) < 2 )
         {
            HPL_pwarn( stderr, __LINE__, "LU_pdinfo",
                       "Value of NDIV less than 2" );
            error = 1; goto label_error;
         }
      }
/*
 * Recursive panel factorization (RF)
 */
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", num ); *NRFS = atoi( num );
      if( ( *NRFS < 1 ) || ( *NRFS > HPL_MAX_PARAM ) )
      {
         HPL_pwarn( stderr, __LINE__, "LU_pdinfo", "%s %s %d",
                    "Number of values of RFACT",
                    "is less than 1 or greater than", HPL_MAX_PARAM );
         error = 1; goto label_error;
      }
      (void) fgets( line, HPL_LINE_MAX - 2, infp ); lineptr = line;
      for( i = 0; i < *NRFS; i++ )
      {
         (void) sscanf( lineptr, "%s", num ); lineptr += strlen( num ) + 1;
         j = atoi( num );
         if(      j == 0 ) RF[ i ] = HPL_LEFT_LOOKING;
         else if( j == 1 ) RF[ i ] = HPL_CROUT;
         else if( j == 2 ) RF[ i ] = HPL_RIGHT_LOOKING;
         else              RF[ i ] = HPL_RIGHT_LOOKING;
      }
/*
 * Broadcast topology (TP) (0=rg, 1=2rg, 2=rgM, 3=2rgM, 4=L)
 */
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", num ); *NTPS = atoi( num );
      if( ( *NTPS < 1 ) || ( *NTPS > HPL_MAX_PARAM ) )
      {
         HPL_pwarn( stderr, __LINE__, "LU_pdinfo", "%s %s %d",
                    "Number of values of BCAST",
                    "is less than 1 or greater than", HPL_MAX_PARAM );
         error = 1; goto label_error;
      }
      (void) fgets( line, HPL_LINE_MAX - 2, infp ); lineptr = line;
      for( i = 0; i < *NTPS; i++ )
      {
         (void) sscanf( lineptr, "%s", num ); lineptr += strlen( num ) + 1;
         j = atoi( num );
         if(      j == 0 ) TP[ i ] = HPL_1RING;
         else if( j == 1 ) TP[ i ] = HPL_1RING_M;
         else if( j == 2 ) TP[ i ] = HPL_2RING;
         else if( j == 3 ) TP[ i ] = HPL_2RING_M;
         else if( j == 4 ) TP[ i ] = HPL_BLONG;
         else if( j == 5 ) TP[ i ] = HPL_BLONG_M;
         else              TP[ i ] = HPL_1RING_M;
      }
/*
 * Lookahead depth (>=0) (NDH)
 */
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", num ); *NDHS = atoi( num );
      if( ( *NDHS < 1 ) || ( *NDHS > HPL_MAX_PARAM ) )
      {
         HPL_pwarn( stderr, __LINE__, "LU_pdinfo", "%s %s %d",
                    "Number of values of DEPTH",
                    "is less than 1 or greater than", HPL_MAX_PARAM );
         error = 1; goto label_error;
      }
      (void) fgets( line, HPL_LINE_MAX - 2, infp ); lineptr = line;
      for( i = 0; i < *NDHS; i++ )
      {
         (void) sscanf( lineptr, "%s", num );
         lineptr += strlen( num ) + 1;
         if( ( DH[ i ] = atoi( num ) ) < 0 )
         {
            HPL_pwarn( stderr, __LINE__, "LU_pdinfo",
                       "Value of DEPTH less than 0" );
            error = 1; goto label_error;
         }
      }
/*
 * Swapping algorithm (0,1 or 2) (FSWAP)
 */
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", num ); j = atoi( num );
      if(      j == 0 ) *FSWAP = HPL_SWAP00;
      else if( j == 1 ) *FSWAP = HPL_SWAP01;
      else if( j == 2 ) *FSWAP = HPL_SW_MIX;
      else              *FSWAP = HPL_SWAP01;
/*
 * Swapping threshold (>=0) (TSWAP)
 */
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", num ); *TSWAP = atoi( num );
      if( *TSWAP <= 0 ) *TSWAP = 0;
/*
 * L1 in (no-)transposed form (0 or 1)
 */
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", num ); *L1NOTRAN = atoi( num );
      if( ( *L1NOTRAN != 0 ) && ( *L1NOTRAN != 1 ) ) *L1NOTRAN = 0; 
/*
 * U  in (no-)transposed form (0 or 1)
 */
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", num ); *UNOTRAN = atoi( num );
      if( ( *UNOTRAN != 0 ) && ( *UNOTRAN != 1 ) ) *UNOTRAN = 0;
/*
 * Equilibration (0=no, 1=yes)
 */
      (void) fgets( line, HPL_LINE_MAX - 2, infp );
      (void) sscanf( line, "%s", num ); *EQUIL = atoi( num );
      if( ( *EQUIL != 0 ) && ( *EQUIL != 1 ) ) *EQUIL = 1;

/*
 * Close input file
 */
label_error:
      (void) fclose( infp );
   }
   else { TEST->outfp = NULL; }
/*
 * Check for error on reading input file
 */
   (void) HPL_all_reduce( (void *)(&error), 1, HPL_INT, HPL_max,
                          MPI_COMM_WORLD );
   if( error )
   {
      if( rank == 0 )
         HPL_pwarn( stderr, __LINE__, "LU_pdinfo",
                    "Illegal input in file HPL.dat. Exiting ..." );
      MPI_Finalize();
#ifdef HPL_CALL_VSIPL
      (void) vsip_finalize( NULL );
#endif
      exit( 1 );
   }
/*
 * Compute and broadcast machine epsilon
 */
   TEST->epsil = HPL_pdlamch( MPI_COMM_WORLD, HPL_MACH_EPS );
/*
 * Pack information arrays and broadcast
 */
   (void) HPL_broadcast( (void *)(&(TEST->thrsh)), 1, HPL_DOUBLE, 0,
                         MPI_COMM_WORLD );
/*
 * Broadcast array sizes
 */
   iwork = (int *)malloc( (size_t)(15) * sizeof( int ) );
   if( rank == 0 )
   {
      iwork[ 0] = *NS;      iwork[ 1] = *NBS;
      iwork[ 2] = ( *PMAPPIN == HPL_ROW_MAJOR ? 0 : 1 );
      iwork[ 3] = *NPQS;    iwork[ 4] = *NPFS;     iwork[ 5] = *NBMS;
      iwork[ 6] = *NDVS;    iwork[ 7] = *NRFS;     iwork[ 8] = *NTPS;
      iwork[ 9] = *NDHS;    iwork[10] = *TSWAP;    iwork[11] = *L1NOTRAN;
      iwork[12] = *UNOTRAN; iwork[13] = *EQUIL;    iwork[14] = *ALIGN;
   }
   (void) HPL_broadcast( (void *)iwork, 15, HPL_INT, 0, MPI_COMM_WORLD );
   if( rank != 0 )
   {
      *NS       = iwork[ 0]; *NBS   = iwork[ 1];
      *PMAPPIN  = ( iwork[ 2] == 0 ?  HPL_ROW_MAJOR : HPL_COLUMN_MAJOR );
      *NPQS     = iwork[ 3]; *NPFS  = iwork[ 4]; *NBMS     = iwork[ 5];
      *NDVS     = iwork[ 6]; *NRFS  = iwork[ 7]; *NTPS     = iwork[ 8];
      *NDHS     = iwork[ 9]; *TSWAP = iwork[10]; *L1NOTRAN = iwork[11];
      *UNOTRAN  = iwork[12]; *EQUIL = iwork[13]; *ALIGN    = iwork[14];
   }
   if( iwork ) free( iwork );
/*
 * Pack information arrays and broadcast
 */
   lwork = (*NS) + (*NBS) + 2 * (*NPQS) + (*NPFS) + (*NBMS) + 
           (*NDVS) + (*NRFS) + (*NTPS) + (*NDHS) + 1;
   iwork = (int *)malloc( (size_t)(lwork) * sizeof( int ) );
   if( rank == 0 )
   {
      j = 0;
      for( i = 0; i < *NS;   i++ ) { iwork[j] = N [i]; j++; }
      for( i = 0; i < *NBS;  i++ ) { iwork[j] = NB[i]; j++; }
      for( i = 0; i < *NPQS; i++ ) { iwork[j] = P [i]; j++; }
      for( i = 0; i < *NPQS; i++ ) { iwork[j] = Q [i]; j++; }
      for( i = 0; i < *NPFS; i++ )
      {
         if(      PF[i] == HPL_LEFT_LOOKING  ) iwork[j] = 0;
         else if( PF[i] == HPL_CROUT         ) iwork[j] = 1;
         else if( PF[i] == HPL_RIGHT_LOOKING ) iwork[j] = 2;
         j++;
      }
      for( i = 0; i < *NBMS; i++ ) { iwork[j] = NBM[i]; j++; }
      for( i = 0; i < *NDVS; i++ ) { iwork[j] = NDV[i]; j++; }
      for( i = 0; i < *NRFS; i++ )
      {
         if(      RF[i] == HPL_LEFT_LOOKING  ) iwork[j] = 0;
         else if( RF[i] == HPL_CROUT         ) iwork[j] = 1;
         else if( RF[i] == HPL_RIGHT_LOOKING ) iwork[j] = 2;
         j++;
      }
      for( i = 0; i < *NTPS; i++ )
      {
         if(      TP[i] == HPL_1RING   ) iwork[j] = 0;
         else if( TP[i] == HPL_1RING_M ) iwork[j] = 1;
         else if( TP[i] == HPL_2RING   ) iwork[j] = 2;
         else if( TP[i] == HPL_2RING_M ) iwork[j] = 3;
         else if( TP[i] == HPL_BLONG   ) iwork[j] = 4;
         else if( TP[i] == HPL_BLONG_M ) iwork[j] = 5;
         j++;
      }
      for( i = 0; i < *NDHS; i++ ) { iwork[j] = DH[i]; j++; }

      if(      *FSWAP == HPL_SWAP00 ) iwork[j] = 0;
      else if( *FSWAP == HPL_SWAP01 ) iwork[j] = 1;
      else if( *FSWAP == HPL_SW_MIX ) iwork[j] = 2;
      j++;
   }
   (void) HPL_broadcast( (void*)iwork, lwork, HPL_INT, 0,
                         MPI_COMM_WORLD );
   if( rank != 0 )
   {
      j = 0;
      for( i = 0; i < *NS;   i++ ) { N [i] = iwork[j]; j++; }
      for( i = 0; i < *NBS;  i++ ) { NB[i] = iwork[j]; j++; }
      for( i = 0; i < *NPQS; i++ ) { P [i] = iwork[j]; j++; }
      for( i = 0; i < *NPQS; i++ ) { Q [i] = iwork[j]; j++; }

      for( i = 0; i < *NPFS; i++ )
      {
         if(      iwork[j] == 0 ) PF[i] = HPL_LEFT_LOOKING;
         else if( iwork[j] == 1 ) PF[i] = HPL_CROUT;
         else if( iwork[j] == 2 ) PF[i] = HPL_RIGHT_LOOKING;
         j++;
      }
      for( i = 0; i < *NBMS; i++ ) { NBM[i] = iwork[j]; j++; }
      for( i = 0; i < *NDVS; i++ ) { NDV[i] = iwork[j]; j++; }
      for( i = 0; i < *NRFS; i++ )
      {
         if(      iwork[j] == 0 ) RF[i] = HPL_LEFT_LOOKING;
         else if( iwork[j] == 1 ) RF[i] = HPL_CROUT;
         else if( iwork[j] == 2 ) RF[i] = HPL_RIGHT_LOOKING;
         j++;
      }
      for( i = 0; i < *NTPS; i++ )
      {
         if(      iwork[j] == 0 ) TP[i] = HPL_1RING;
         else if( iwork[j] == 1 ) TP[i] = HPL_1RING_M;
         else if( iwork[j] == 2 ) TP[i] = HPL_2RING;
         else if( iwork[j] == 3 ) TP[i] = HPL_2RING_M;
         else if( iwork[j] == 4 ) TP[i] = HPL_BLONG;
         else if( iwork[j] == 5 ) TP[i] = HPL_BLONG_M;
         j++;
      }
      for( i = 0; i < *NDHS; i++ ) { DH[i] = iwork[j]; j++; }

      if(      iwork[j] == 0 ) *FSWAP = HPL_SWAP00;
      else if( iwork[j] == 1 ) *FSWAP = HPL_SWAP01;
      else if( iwork[j] == 2 ) *FSWAP = HPL_SW_MIX;
      j++;
   }
   if( iwork ) free( iwork );

/*
 * End of LU_pdinfo
 */
}