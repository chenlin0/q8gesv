#!/bin/bash
rm *.o
rm *.a
rm test_gemm
gcc -c  convert.c matgen.c print.c q8gesv.c isCorrect.c hpl_ai_gesv.c gmres.c q8gemm.c timer.c -O3 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -lmkl_rt -mfma -mavx2  -DHPL_CALL_MKL 
ar crv libhpl_ai.a  *.o
gcc -o test_gemm test_gemm.c -L. -lhpl_ai -O3 -I/opt/intel/mkl/include -L/opt/intel/mkl/lib/intel64 -lmkl_rt -mfma -mavx2 -lm -DHPL_CALL_MKL 
for((n=2000;n<=20000;n=n+2000))
do
    ./test_gemm $n
done