echo "This can only work for galois 2.2.1 standalone converter"

declare -arr graph=(facebook europe.osm kron24 livejournal orkut pokec random roadNet-CA rmat uk2002 twitter)

path=/mnt/raid0_huge/hang/simdx-dataset/
binpath=/home/hang/Galois-2.2.1/build/release/tools/graph-convert-standalone


for file in ${graph[@]};
do
	#$binpath/graph-convert-standalone -edgelist2vgr $path/$file/$file.edgelist $path/$file/$file.gr
	$binpath/graph-convert-standalone -gr2randintgr -maxValue=64 $path/$file/$file.gr $path/$file/$file.weighted.gr
done
