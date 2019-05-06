//10/03/2016
//Graph data structure on GPUs
#ifndef _GPU_GRAPH_H_
#define _GPU_GRAPH_H_

#include "header.h"
#include "util.h"
#include "graph.h"

class gpu_graph
{
	public:
		vertex_t *adj_list;
		weight_t *weight_list;
		index_t *beg_pos;
		
		index_t vert_count;
		index_t edge_count;
		index_t avg_degree;

	public:
		~gpu_graph(){}
		
		gpu_graph(
			graph<long, long, long, vertex_t, index_t, weight_t> *ginst)
		{
			vert_count=ginst->vert_count;
			edge_count=ginst->edge_count;
			avg_degree = ginst->edge_count/ginst->vert_count;

			size_t weight_sz=sizeof(weight_t)*edge_count;
			size_t adj_sz=sizeof(vertex_t)*edge_count;
			size_t beg_sz=sizeof(index_t)*(vert_count+1);
			
			/* Alloc GPU space */
			H_ERR(cudaMalloc((void **)&adj_list, adj_sz));
			H_ERR(cudaMalloc((void **)&beg_pos, beg_sz));
			H_ERR(cudaMalloc((void **)&weight_list, weight_sz));
			
			/* copy it to GPU */
			H_ERR(cudaMemcpy(adj_list,ginst->adj_list,
						adj_sz, cudaMemcpyHostToDevice));
			H_ERR(cudaMemcpy(beg_pos,ginst->beg_pos,
						beg_sz, cudaMemcpyHostToDevice));
			
			H_ERR(cudaMemcpy(weight_list,ginst->weight,
						weight_sz, cudaMemcpyHostToDevice));
		}
};

#endif
