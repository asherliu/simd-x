declare -arr graph=(facebook europe.osm kron24 livejournal orkut pokec random roadNet-CA rmat uk2002 twitter)

path=/home/hang/raid0_c0_ptr/hang/simdx-dataset/
binpath=/home/hang/ligra/utils


for file in ${graph[@]};
do
	$binpath/SNAPtoAdj $path/$file/$file.edgelist $path/$file/$file.ligra
	$binpath/adjGraphAddWeights $path/$file/$file.ligra $path/$file/$file.weightedligra
done
