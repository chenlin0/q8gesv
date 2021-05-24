#include"LU.h"
int LU_solve(HPL_LU *lu){
/* 
 * Purpose
 * =======
 *
 * read "LU.dat".
 * LU factorization
 *
 * ---------------------------------------------------------------------
 */ 
/*
 */
   int                        nval  [HPL_MAX_PARAM],
                              nbval [HPL_MAX_PARAM],
                              pval  [HPL_MAX_PARAM],
                              qval  [HPL_MAX_PARAM],
                              nbmval[HPL_MAX_PARAM],
                              ndvval[HPL_MAX_PARAM],
                              ndhval[HPL_MAX_PARAM];

   HPL_T_FACT                 pfaval[HPL_MAX_PARAM],
                              rfaval[HPL_MAX_PARAM];

   HPL_T_TOP                  topval[HPL_MAX_PARAM];


   int                        L1notran, Unotran, align, equil, in, inb,
                              inbm, indh, indv, ipfa, ipq, irfa, itop,
                              mycol, myrow, ns, nbs, nbms, ndhs, ndvs,
                              npcol, npfs, npqs, nprow, nrfs, ntps, 
                              rank, size, tswap;
   HPL_T_ORDER                pmapping;
   HPL_T_FACT                 rpfa;
   HPL_T_SWAP                 fswap;
   double t0,t1;
   int count=0;
/* ..
 * .. Executable Statements ..
 */

   MPI_Comm_rank( MPI_COMM_WORLD, &rank );
   MPI_Comm_size( MPI_COMM_WORLD, &size );
   ns = 1;
   nval[0] = lu->N;
   nbs = 1;
   nbval[0] = lu->NB;
   pmapping = 1;
   npqs = 1;
   pval[0] = lu->P;
   qval[0] = lu->Q;
   align = lu->algo.align;

   LU_pdinfo( &(lu->test), &ns, nval, &nbs, nbval, &pmapping, &npqs, pval, qval,
               &npfs, pfaval, &nbms, nbmval, &ndvs, ndvval, &nrfs, rfaval,
               &ntps, topval, &ndhs, ndhval, &fswap, &tswap, &L1notran,
               &Unotran, &equil, &align );
/*
 * Loop over different process grids - Define process grid. Go to bottom
 * of process grid loop if this case does not use my process.
 */
    lu->nTest = npqs*ns*nbs*ndhs*ntps*\
    nrfs*npfs*nbms*ndvs;
    lu->gflops = (double*)malloc(lu->nTest*sizeof(double));
      (void) HPL_grid_info( &(lu->grid), &nprow, &npcol, &myrow, &mycol );
      if( ( myrow < 0 ) || ( myrow >= nprow ) ||
          ( mycol < 0 ) || ( mycol >= npcol ) ){
             if(myrow ==0 && mycol ==0){
                printf("进程网格P,Q值设计错误\n");
             }
             MPI_Finalize();
             exit(-1);
          }
   count = 0;
   for( ipq = 0; ipq < npqs; ipq++ )
   {

      for( in = 0; in < ns; in++ )
      {                            /* Loop over various problem sizes */
       for( inb = 0; inb < nbs; inb++ )
       {                        /* Loop over various blocking factors */
        for( indh = 0; indh < ndhs; indh++ )
        {                       /* Loop over various lookahead depths */
         for( itop = 0; itop < ntps; itop++ )
         {                  /* Loop over various broadcast topologies */
          for( irfa = 0; irfa < nrfs; irfa++ )
          {             /* Loop over various recursive factorizations */
           for( ipfa = 0; ipfa < npfs; ipfa++ )
           {                /* Loop over various panel factorizations */
            for( inbm = 0; inbm < nbms; inbm++ )
            {        /* Loop over various recursive stopping criteria */
             for( indv = 0; indv < ndvs; indv++ )
             {          /* Loop over various # of panels in recursion */
/*
 * Set up the algorithm parameters
 */
              lu->algo.btopo = topval[itop]; 
              lu->algo.depth = ndhval[indh];
              lu->algo.nbmin = nbmval[inbm]; 
              lu->algo.nbdiv = ndvval[indv];

              lu->algo.pfact = rpfa = pfaval[ipfa];

              if( L1notran != 0 )
              {
                 if( rpfa == HPL_LEFT_LOOKING ) lu->algo.pffun = HPL_pdpanllN;
                 else if( rpfa == HPL_CROUT   ) lu->algo.pffun = HPL_pdpancrN;
                 else                           lu->algo.pffun = HPL_pdpanrlN;

                 lu->algo.rfact = rpfa = rfaval[irfa];
                 if( rpfa == HPL_LEFT_LOOKING ) lu->algo.rffun = HPL_pdrpanllN;
                 else if( rpfa == HPL_CROUT   ) lu->algo.rffun = HPL_pdrpancrN;
                 else                           lu->algo.rffun = HPL_pdrpanrlN;

                 if( Unotran != 0 ) lu->algo.upfun = HPL_pdupdateNN;
                 else               lu->algo.upfun = HPL_pdupdateNT;
              }
              else
              {
                 if( rpfa == HPL_LEFT_LOOKING ) lu->algo.pffun = HPL_pdpanllT;
                 else if( rpfa == HPL_CROUT   ) lu->algo.pffun = HPL_pdpancrT;
                 else                           lu->algo.pffun = HPL_pdpanrlT;

                 lu->algo.rfact = rpfa = rfaval[irfa];
                 if( rpfa == HPL_LEFT_LOOKING ) lu->algo.rffun = HPL_pdrpanllT;
                 else if( rpfa == HPL_CROUT   ) lu->algo.rffun = HPL_pdrpancrT;
                 else                           lu->algo.rffun = HPL_pdrpanrlT;

                 if( Unotran != 0 ) lu->algo.upfun = HPL_pdupdateTN;
                 else               lu->algo.upfun = HPL_pdupdateTT;
              }

              lu->algo.fswap = fswap; 
              lu->algo.fsthr = tswap;
              lu->algo.equil = equil; 
              lu->algo.align = align;
              HPL_barrier( lu->grid.all_comm );
              t0 = MPI_Wtime();
              HPL_pdgesv( &(lu->grid), &(lu->algo), &(lu->mat) );
              HPL_barrier( lu->grid.all_comm );
              t1 = MPI_Wtime()-t0;
              if( lu->mat.info != 0 ){
               if(myrow ==0 && mycol ==0){
                  printf("矩阵不满秩\n");
               }
               MPI_Finalize();
               exit(-1);
              }
            #ifdef HPL_AI_COMPLEX_DOUBLE
                lu->gflops[count] = (((double)(lu->N) /   1.0e+9 ) * 
                 ( (double)(lu->N) / t1 ) ) * 
                 ( ( 2.0 ) * (double)(lu->N)  );
            #else
                lu->gflops[count] = ( ( (double)(lu->N) /   1.0e+9 ) * 
                 ( (double)(lu->N) / t1 ) ) * 
                 ( ( 2.0 / 3.0 ) * (double)(lu->N) + ( 3.0 / 2.0 ) );
            #endif
                if(myrow==0 && mycol==0){
                    printf("Rmax=%19.4e GFLOPS\n",lu->gflops[count]);
                }
                count++;
             }
            }
           }
          }
         }
        }
       }
      }
     
   }

   return( 0 );
}