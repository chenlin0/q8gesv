## test the performance of q8gemm  $C_{N*512} - = A_{N*512} * B_{512*N}$
./test_gemm.sh
or
./test_gemm N

## test the performance of q8gesv $A_{N\times N}x_{N}=b_{N}$
./test_gesv.sh
or 
./test_gesv N

## test both
./run.sh

require: Intel MKL 
$mc,nc,kc$ is defined in hpl-ai.h