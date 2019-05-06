folder=ptx
regLimit=64 

#ptxas  -arch=sm_35 -m64 -v -maxrregcount=$regLimit  --generate-line-info "$folder/bfs.ptx"  -o "$folder/bfs.sm_35.cubin" 
#fatbinary --create="$folder/bfs.fatbin" -64 --key="xxxxxxxxxx" --ident="bfs.cu" --cmdline="-v  -maxrregcount=$regLimit  --generate-line-info " "--image=profile=sm_35,file=$folder/bfs.sm_35.cubin" "--image=profile=compute_35,file=$folder/bfs.ptx" --embedded-fatbin="$folder/bfs.fatbin.c" --cuda
gcc -D__CUDA_ARCH__=350 -E -x c++     -DCUDA_DOUBLE_MATH_FUNCTIONS   -Wall -O3 -D__CUDA_PREC_DIV -D__CUDA_PREC_SQRT -I"../../lib/" -I"." "-I/c1/apps/cuda/toolkit/7.5.18/bin/..//include"   -m64 "$folder/bfs.cudafe1.cpp" > "$folder/bfs.cu.cpp.ii" 
gcc -c -x c++ -Wall -O3 -I"../../lib/" -I"." "-I/c1/apps/cuda/toolkit/7.5.18/bin/..//include"   -fpreprocessed -m64 -o "bfs.o" "$folder/bfs.cu.cpp.ii" 
nvlink --arch=sm_35 --register-link-binaries="$folder/bfs_dlink.reg.c" -m64   "-L/c1/apps/cuda/toolkit/7.5.18/bin/..//lib64/stubs" "-L/c1/apps/cuda/toolkit/7.5.18/bin/..//lib64" -cpu-arch=X86_64 "bfs.o"  -lcudadevrt  -o "$folder/bfs_dlink.sm_35.cubin"
fatbinary --create="$folder/bfs_dlink.fatbin" -64 --key="bfs_dlink" --ident="bfs.o " --cmdline="-v   -maxrregcount=$regLimit --generate-line-info " -link "--image=profile=sm_35,file=$folder/bfs_dlink.sm_35.cubin" --embedded-fatbin="$folder/bfs_dlink.fatbin.c" 
gcc -c -x c++ -DFATBINFILE="\"$folder/bfs_dlink.fatbin.c\"" -DREGISTERLINKBINARYFILE="\"$folder/bfs_dlink.reg.c\"" -I. -Wall -O3 -I"../../lib/" -I"." "-I/c1/apps/cuda/toolkit/7.5.18/bin/..//include"   -D"__CUDACC_VER__=70517" -D"__CUDACC_VER_BUILD__=17" -D"__CUDACC_VER_MINOR__=5" -D"__CUDACC_VER_MAJOR__=7" -m64 -o "$folder/bfs_dlink.o" "/c1/apps/cuda/toolkit/7.5.18/bin/crt/link.stub" 
g++ -Wall -O3 -m64 -o "bfs.bin" -Wl,--start-group "$folder/bfs_dlink.o" "bfs.o"   "-L/c1/apps/cuda/toolkit/7.5.18/bin/..//lib64/stubs" "-L/c1/apps/cuda/toolkit/7.5.18/bin/..//lib64" -lcudadevrt  -lcudart_static  -lrt -lpthread  -ldl  -Wl,--end-group 
#
