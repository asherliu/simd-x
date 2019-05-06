#declare -arr graph=(facebook europe.osm kron24 livejournal orkut pokec random roadNet-CA rmat uk2002 twitter)
#declare -arr bfspull=(3 0   2   4   3   3   3   0   2   4   2);
#declare -arr sssppull=(4    0   3   4   4   5   5   0   2   3   2);

declare -arr graph=(facebook kron24 livejournal orkut pokec random rmat uk2002 twitter)
declare -arr bfspull=(3		2	4	3	3	3	2	4	2);
#declare -arr sssppull=(3		2	4	3	3	4		1	3	2);
declare -arr sssppull=(2	1	2	2	2	2	1	2	2);

#declare -arr graph=(europe.osm roadNet-CA)
#declare -arr bfspull=(0 0);
#declare -arr sssppull=(0 0);

path=/mnt/raid0_huge/hang/simdx_eval/simdx-dataset/

src=1
iter_limit=3
pull_count=2
blk=128

ptr=0;

for file in ${graph[@]};
do
    data=$(echo $path/"$file"/"$file"_beg_pos.bin $path/"$file"/"$file"_csr.bin $path/"$file"/"$file"_weight.bin)
    echo $data
    
    bfsp=${bfspull[$ptr]};
    ssspp=${sssppull[$ptr]};
    
    echo $bfsp, $ssspp;

    #./bfs_low_diameter_optimal.bin $data $src $blk $bfsp  &>> bfs_low_diameter_optimal_"$file".log
    #./bfs_high_diameter_optimal.bin $data $src $blk $bfsp  &>> bfs_highdiameter_optimal_"$file".log
    #./bfs_low_diameter.bin  $data $src $blk $bfsp &>> bfs_optimal_"$file".log
    #./bp_optimal.bin $data $iter_limit $blk &>> bp_optimal_"$file".log
    #./kcore_optimal.bin $data $blk &>> kcore_optimal_"$file".log
    #./pagerank_optimal.bin $data $iter_limit $blk &>> pagerank_optimal_"$file".log
    #./spmv_optimal.bin $data $blk &>> spmv_optimal_"$file".log
    #./sssp_optimal.bin $data $src $blk  $ssspp &>> sssp_optimal_"$file".log
    #echo ./sssp_low_diameter_optimal.bin $data $src $blk  $ssspp
    ./sssp_low_diameter_optimal.bin $data $src $blk  $ssspp &>> sssp_low_diameter_optimal_"$file".log

    ptr=$((ptr + 1))
done

