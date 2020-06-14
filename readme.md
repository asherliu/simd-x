-----
Software requirement
-----
gcc 4.4.7 or higher 

CUDA 7.5 or higher 

-----
Hardware
------
GPU: K20, K40, P100, V100 (tested)
> Generally, the GPU needs to support shuffle instructions.

-----
Compile
-----
nvcc 7.5 or higher

cd simd-x/test/optimal/``app``
-type ``make``

> For instance, one can enter ``simd-x/test/optimal/bfs_high_diameter/`` and type ``make``


-----
Execute
------

For each application, once you type the compiled executable, the binary file will remind you the files of interest. 

> Using ``bfs_high_diameter`` as an example, one will need ``/path/to/exe /path/to/beg_pos /path/to/adj_list /path/weight_list src blk_size swith_iter`` to execute the file. Below are the explanation of each parameter:
> - `path/to/exe`: the path to this executable.
> - `/path/to/beg_pos`: the path to the begin_position array of the graph dataset. We explained the begin position file [here](https://github.com/asherliu/graph_project_start/blob/master/README.md).
> Similarly for `/path/to/adj_list` and `/path/weight_list`. It is important to note that, for applications that do not need weight file (such as bfs), we can provide an invalid path to the `/path/weight_list` parameter.
> - `src` stands for where the users want the BFS starts.
> - `blk_size` means the number of thread blocks we want the kernel to have.  
> - `swith_iter` means which iteration to switch the BFS direction from top-down to bottom-up. 



**Should you have any questions about this project, please contact us by asher.hangliu@gmail.com.**

-----
Reference
-------
   [USENIX ATC '19] SIMD-X: Programming and Processing of Graph Algorithms on GPUs [[PDF](https://arxiv.org/pdf/1812.04070.pdf)]


