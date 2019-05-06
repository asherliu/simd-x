#ifndef _H_META_DATA_
#define _H_META_DATA_

#include "header.h"
#include "util.h"
//#include "mapper_enactor.cuh"

class meta_data 
{
	public:
		/*traversal*/
		feature_t *vert_status;
		feature_t *vert_status_prev;
		
		bit_t *bitmap;
	
		//For debug
		feature_t *sa_chk;
		index_t *sml_count_chk;
		index_t *mid_count_chk;
		index_t *lrg_count_chk;
	
		index_t *future_work;

		/*reducer*/
		index_t *cat_thd_count_sml;
		index_t *cat_thd_count_mid;
		index_t *cat_thd_count_lrg;
		index_t *cat_thd_off_sml;
		index_t *cat_thd_off_mid;
		index_t *cat_thd_off_lrg;
		
		index_t *cat_thd_count_h;
		index_t *cat_thd_off_h;
		index_t *scan_temp_sml;//store block sum
		index_t *scan_temp_mid;//store block sum
		index_t *scan_temp_lrg;//store block sum

		/*worklist*/
		vertex_t *worklist_sml;
		vertex_t *worklist_mid;
		vertex_t *worklist_lrg;
		//vertex_t *worklist_sz_sml_d;
		//vertex_t *worklist_sz_mid_d;
		//vertex_t *worklist_sz_lrg_d;

		/*thread bin
		 * - for generate frontier queue while map
		 *   instead of scan and generate frontier queue
		 * */
		vertex_t *worklist_bin;
		volatile vertex_t *worklist_sz_sml;
		volatile vertex_t *worklist_sz_mid;
		volatile vertex_t *worklist_sz_lrg;

		//volatile vertex_t *worklist_sz_sml;
		//volatile vertex_t *worklist_sz_mid;
		//volatile vertex_t *worklist_sz_lrg;

		/*stream*/
		cudaStream_t *stream;
	
	public:
		~meta_data(){}
		meta_data(
			vertex_t vert_count,
			index_t edge_count)
		{
			const size_t VERT_SZ=sizeof(vertex_t)*vert_count;
			const size_t FEAT_SZ=sizeof(feature_t)*vert_count;
			const size_t BIT_SZ=sizeof(bit_t)*(((vert_count)>>3) + 1);
			const size_t CATE_SZ=sizeof(index_t)*BLKS_NUM*THDS_NUM;

			//int blk_size = 256;
			//int grd_size = 256;
	

			//Because thread bin is only used by the smaller kernel
			//cudaOccupancyMaxPotentialBlockSize(&grd_size, &blk_size, 
			//	mapper_merge_push_kernel, 0, 0);
			const size_t GBIN_SZ=sizeof(vertex_t)*BLKS_NUM*THDS_NUM*BIN_SZ;
			
			assert(THDS_NUM <= 1024);
			H_ERR(cudaMalloc((void **)&vert_status, FEAT_SZ));
			H_ERR(cudaMalloc((void **)&vert_status_prev, FEAT_SZ));
			H_ERR(cudaMalloc((void **)&bitmap, BIT_SZ));
			H_ERR(cudaMemset(bitmap, 0, BIT_SZ));	

			//Thread-bin for frontier generation
			H_ERR(cudaMalloc((void **)&worklist_bin,GBIN_SZ));

			//H_ERR(cudaMalloc((void **)&worklist_sz, sizeof(vertex_t)));
			//Workload balancing: count, prefix and collect.
			H_ERR(cudaMalloc((void **)&worklist_sml,VERT_SZ));
			H_ERR(cudaMalloc((void **)&worklist_mid,VERT_SZ * 16));
			H_ERR(cudaMalloc((void **)&worklist_lrg,VERT_SZ));
			H_ERR(cudaMalloc((void **)&cat_thd_count_sml,CATE_SZ));
			H_ERR(cudaMalloc((void **)&cat_thd_count_mid,CATE_SZ));
			H_ERR(cudaMalloc((void **)&cat_thd_count_lrg,CATE_SZ));
			H_ERR(cudaMalloc((void **)&cat_thd_off_sml,CATE_SZ));
			H_ERR(cudaMalloc((void **)&cat_thd_off_mid,CATE_SZ));
			H_ERR(cudaMalloc((void **)&cat_thd_off_lrg,CATE_SZ));

			//maybe should be size of BLKS_NUM only!
			H_ERR(cudaMalloc((void **)&scan_temp_sml,CATE_SZ));
			H_ERR(cudaMalloc((void **)&scan_temp_mid,CATE_SZ));
			H_ERR(cudaMalloc((void **)&scan_temp_lrg,CATE_SZ));

			//verification purpose
			H_ERR(cudaMallocHost((void **)&cat_thd_count_h,CATE_SZ));
			H_ERR(cudaMallocHost((void **)&cat_thd_off_h,CATE_SZ));

			//maybe should be size of BLKS_NUM only!
			H_ERR(cudaMalloc((void **)&worklist_sz_sml,sizeof(vertex_t)));
			H_ERR(cudaMalloc((void **)&worklist_sz_mid,sizeof(vertex_t)));
			H_ERR(cudaMalloc((void **)&worklist_sz_lrg,sizeof(vertex_t)));
			
			//for frontier queue size report
			//Allocated on CPU, mapped to GPU
			//H_ERR(cudaHostAlloc((void **)&worklist_sz_sml,
			//			sizeof(index_t),cudaHostAllocMapped));
			//H_ERR(cudaHostGetDevicePointer((void **)&worklist_sz_sml_d,
			//			worklist_sz_sml,0));
			//H_ERR(cudaHostAlloc((void **)&worklist_sz_mid,
			//			sizeof(index_t),cudaHostAllocMapped));
			//H_ERR(cudaHostGetDevicePointer((void **)&worklist_sz_mid_d,
			//			worklist_sz_mid,0));
			//H_ERR(cudaHostAlloc((void **)&worklist_sz_lrg,
			//			sizeof(index_t),cudaHostAllocMapped));
			//H_ERR(cudaHostGetDevicePointer((void **)&worklist_sz_lrg_d,
			//			worklist_sz_lrg,0));

			stream = (cudaStream_t *)malloc(sizeof(cudaStream_t)*3);
			for(index_t i=0;i<3;++i)
				H_ERR(cudaStreamCreate(&(stream[i])));
		

			H_ERR(cudaMallocHost((void **)&sa_chk, FEAT_SZ));
			H_ERR(cudaMallocHost((void **)&sml_count_chk, CATE_SZ));
			H_ERR(cudaMallocHost((void **)&mid_count_chk, CATE_SZ));
			H_ERR(cudaMallocHost((void **)&lrg_count_chk, CATE_SZ));

			H_ERR(cudaMalloc((void **)&future_work, sizeof(index_t)));
		}
};

#endif
