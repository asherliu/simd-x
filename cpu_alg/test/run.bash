declare -arr graph=(facebook europe.osm kron24 livejournal orkut pokec random roadNet-CA rmat uk2002 twitter)

#exe=bfs.bin
#exe=test.bin
exe=kcore.bin
#path=/mnt/raid0_huge/hang/discovery_dataset
path=/lustre/groups/huanglab/discovery_dataset/
for file in ${graph[@]};
do

	echo
	echo
	echo ./$exe $path/"$file"/"$file"_beg_pos.bin $path/"$file"/"$file"_csr.bin $path/"$file"/"$file"_weight.bin 96
	./$exe $path/"$file"/"$file"_beg_pos.bin $path/"$file"/"$file"_csr.bin $path/"$file"/"$file"_weight.bin 96 
done

