#ifndef __REDUCER_H__
#define __REDUCER_H__

#include "util.h"
#include "header.h"
#include "prefix_scan.cuh"
#include "meta_data.cuh"
#include "gpu_graph.cuh"
#include <limits.h>
#include <assert.h>

__global__ void 
thread_stride_gather(
		feature_t level,
		gpu_graph ggraph,
		meta_data mdata,
		reducer reducer_inst
){
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	//const index_t GRNTY = blockDim.x * gridDim.x;
	//const index_t WOFF = threadIdx.x & 31;
	//const index_t wid_in_blk = threadIdx.x >> 5;
	//const index_t wcount_in_blk = blockDim.x >> 5;
	const index_t BIN_OFF = TID * BIN_SZ;

	//worklist_gather._thread_stride_gather
	//		(mdata.worklist_mid, 
	//		 mdata.worklist_bin, 
	//		 my_front_count, 
	//		 output_off, 
	//		 BIN_OFF);

	reducer_inst._thread_stride_gather(
	mdata.worklist_mid, mdata.worklist_bin, mdata.cat_thd_count_mid[TID],
    mdata.cat_thd_off_mid[TID], BIN_OFF);
}


/* Scan status array to generate *sorted* frontier queue */
__global__ void 
gen_push_worklist(
		feature_t level,
		gpu_graph ggraph,
		meta_data mdata,
		reducer reducer_inst
){
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t WOFF = threadIdx.x & 31;
	const index_t wid_in_blk = threadIdx.x >> 5;
	const index_t wcount_in_blk = blockDim.x >> 5;

	reducer_inst._push_coalesced_scan_random_list(
	TID, wid_in_blk, WOFF, wcount_in_blk, GRNTY,	level);
}


__global__ void 
gen_pull_worklist(
		feature_t level,
		gpu_graph ggraph,
		meta_data mdata,
		reducer reducer_inst
){
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t WOFF = threadIdx.x & 31;
	const index_t wid_in_blk = threadIdx.x >> 5;
	const index_t wcount_in_blk = blockDim.x >> 5;

	reducer_inst._pull_coalesced_scan_sorted_list(
	TID, wid_in_blk, WOFF, wcount_in_blk, GRNTY,	level);
}

__global__ void 
gen_pull_strided_worklist(
		feature_t level,
		gpu_graph ggraph,
		meta_data mdata,
		reducer reducer_inst
){
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t WOFF = threadIdx.x & 31;
	const index_t wid_in_blk = threadIdx.x >> 5;
	const index_t wcount_in_blk = blockDim.x >> 5;

	reducer_inst._pull_strided_scan_sorted_list(
	TID, wid_in_blk, WOFF, wcount_in_blk, GRNTY,	level);
}


/* Call by users */
int reducer_push
(feature_t level,
 gpu_graph ggraph,
 meta_data mdata,
 reducer worklist_gather
 ){
	gen_push_worklist<<<BLKS_NUM, THDS_NUM>>>
		(level, ggraph, mdata, worklist_gather);
	H_ERR(cudaThreadSynchronize());
	return 0;
}

int reducer_pull
(feature_t level,
 gpu_graph ggraph,
 meta_data mdata,
 reducer worklist_gather
 ){
	gen_pull_worklist<<<BLKS_NUM, THDS_NUM>>>
		(level, ggraph, mdata, worklist_gather);
	H_ERR(cudaThreadSynchronize());
	return 0;
}

int reducer_strided_pull
(feature_t level,
 gpu_graph ggraph,
 meta_data mdata,
 reducer worklist_gather
 ){
	gen_pull_strided_worklist<<<BLKS_NUM, THDS_NUM>>>
		(level, ggraph, mdata, worklist_gather);
	H_ERR(cudaThreadSynchronize());
	return 0;
}

#endif
