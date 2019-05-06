folder="ptx"

ptxas  -arch=sm_35 -m64 -v -v  --generate-line-info "$folder/bfs.ptx"  -o "$folder/bfs.sm_35.cubin" 
fatbinary --create="$folder/bfs.fatbin" -64 --ident="bfs.cu" --cmdline="-v  --generate-line-info " "--image=profile=sm_35,file=$folder/bfs.sm_35.cubin" "--image=profile=compute_35,file=$folder/bfs.ptx" --embedded-fatbin="$folder/bfs.fatbin.c" --cuda
clang -D__CUDA_ARCH__=350 -E -x c++ -DCUDA_DOUBLE_MATH_FUNCTIONS  -D__USE_FAST_MATH__=0  -Wall -D__CUDA_FTZ=0 -D__CUDA_PREC_DIV=1 -D__CUDA_PREC_SQRT=1 -I"../../lib/" -I"." -I"../../cpu_alg/" "-I/Developer/NVIDIA/CUDA-8.0/bin//../include"   -m64 "$folder/bfs.cudafe1.cpp" > "$folder/bfs.cu.cpp.ii" 
clang -c -x c++ -Wall -I"../../lib/" -I"." -I"../../cpu_alg/" "-I/Developer/NVIDIA/CUDA-8.0/bin//../include"   -m64 -o "bfs.o" "$folder/bfs.cu.cpp.ii"
