#declare -arr folder=(apsp  bfs  kcore pagerank  spmv wcc wcc_hybrid)
declare -arr folder=(bfs_high_diameter  bfs_low_diameter  bp  kcore pagerank  spmv  sssp_high_diameter sssp_low_diameter)
#for dir in ${folder[@]}
for dir in ${folder[@]}
do
	echo $dir
	cd $dir
	#sed -i 's/-DBLK_SZ=512/-DBLK_SZ=4096/g' Makefile
	#sed -i 's/-DBLK_SZ=4096/-DBLK_SZ=512/g' Makefile
	make clean
	make
	#make #time=1 profile=1
	#file=$(basename $(find . -name "aio*.bin") | sed -e 's/\.bin/_b512.bin/g')
	file=$(basename $(find . -name "*bin") | sed -e 's/\.bin/_p100.bin/g')
	#file=$(basename $(find . -name "aio*.bin") | sed -e 's/\.bin/_iolist.bin/g')
	#echo $file
	#cp aio*.bin /mnt1/hang/evaluations/graphene_nosort/workload_stealing/$file 
	#echo cp *.bin /mnt/raid0_huge/hang/simdx-test/ballot-online-activation-pattern/$file 
	echo cp *.bin /mnt/raid0_huge/hang/simdx-test/ballot-online-activation-pattern/$file 
	#cp *.bin /mnt/raid0_huge/hang/simdx-test/ballot-online-activation-pattern/$file 
	cp *.bin /home/hangliu/simdx-test/p100/$file 
	#mv ../test/aio_"$dir"_io_hp_sort ../test/aio_"$dir"_io_hp_compress_sort

	cd ..
done
