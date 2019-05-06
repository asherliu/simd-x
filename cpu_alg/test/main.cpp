#include "cpu_pagerank.hpp"
#include "cpu_delta_pagerank.hpp"
#include "cpu_kcore.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "graph.h"

int main(int args, char **argv)
{
	std::cout<<"Input: /path/to/kcore /path/to/beg_pos /path/to/adj_list /path/weight_list k\n";
	if(args<5){std::cout<<"Wrong input\n";exit(-1);}
		
	double tm_map,tm_red,tm_scan;
	char *file_beg_pos = argv[1];
	char *file_adj_list = argv[2];
	char *file_weight_list = argv[3];
	vertex_t k = (vertex_t)atol(argv[4]);
	
	//Read graph to CPU
	graph<long, long, long,vertex_t, index_t, weight_t>
	*ginst=new graph<long, long, long,vertex_t, index_t, weight_t>
	(file_beg_pos, file_adj_list, file_weight_list);
    
    index_t kcore_count = 0;
    //!!!TODO
    //also need to change the Include path in Makefile
    feature_t *dist;
    //cpu_pagerank<index_t, vertex_t, feature_t>
    //    (dist, ginst->vert_count, ginst->edge_count, ginst->beg_pos, ginst->adj_list);
    //cpu_delta_pagerank<index_t, vertex_t, feature_t>
    //    (dist, ginst->vert_count, ginst->edge_count, ginst->beg_pos, ginst->adj_list);
    cpu_kcore<index_t, vertex_t, feature_t>
        (kcore_count, dist, k, ginst->vert_count, ginst->edge_count, ginst->beg_pos, ginst->adj_list);
    return 0;
}
