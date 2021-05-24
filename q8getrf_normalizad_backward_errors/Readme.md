require: Intel MKL 
$mc,nc,kc$ is defined in hpl-ai.h
Suppose $C_{M\times N} -= A_{M\times K}*B_{K\times N}$.
when K is equal to valueKq8, HPL_dgemm call q8gemm.
Otherwise, HPL_dgemm call cblas_sgemm.
Now, we choose ```#define valueKq8 512``` and NB=512.

# Compile
Firstly, please modify TOPdir in Make.Linux
Then ./compile.sh

It is the same as the HPL benchmark.
