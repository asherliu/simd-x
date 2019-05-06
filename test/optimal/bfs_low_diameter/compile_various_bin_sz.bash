for ((i=4; i <= 8192; i=i*2))
do
    echo --------------
    echo "i = $i"
    echo -------------------

    sed -i "s/#define BIN_SZ.*/#define BIN_SZ $i/g" header.h
    cat header.h

    echo
    echo
    echo

    make clean

    make 

    newname="bfs_optimal_low_diameter_bin_sz_"$i".bin"
    mv bfs_optimal_low_diameter.bin /home/hang/simdx-test/optimal-titan-diff-bin-size/$newname

done
