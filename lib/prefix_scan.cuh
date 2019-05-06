#include <stdio.h>
#include <iostream>
#include <assert.h>

#ifndef __PREFIX_SCAN__
#define __PREFIX_SCAN__

#define NUM_BANKS           32
#define LOG_NUM_BANKS       5

#define CONFLICT_FREE_OFFSET(n) \
	((n) >>LOG_NUM_BANKS)



template<typename data_t, typename index_t>
__forceinline__ __device__ void warp_scan_inc(data_t &value, const index_t laneId) 
{
	//kogge-stone scan method
	value +=__shfl_up(value, 1, 32)*(laneId >= 1);
	value +=__shfl_up(value, 2, 32)*(laneId >= 2);
	value +=__shfl_up(value, 4, 32)*(laneId >= 4);
	value +=__shfl_up(value, 8, 32)*(laneId >= 8);
	value +=__shfl_up(value, 16, 32)*(laneId >= 16);
}

template<typename data_t, typename index_t>
__forceinline__ __device__ void warp_scan_exc(data_t &value, const index_t laneId) 
{
	data_t init=value;
	warp_scan_inc(value, laneId);
	value -=init;
}

/* makes sure one warp can do CTA scan
 * One func do prefix scan for three values: sml, mid, lrg
 */
template<typename data_t, typename index_t>
__forceinline__ __device__ void _grid_sum(
		const data_t input, 
		index_t *total
){
	__shared__ data_t smem[1024];
	assert(blockDim.x <= 1024);
	smem[threadIdx.x] = input;
	__syncthreads();

//	if (blockDim.x >= 1024) {if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512]; __syncthreads();}
//	if (blockDim.x >= 512) {if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256]; __syncthreads();}
//	if (blockDim.x >= 256) {if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128]; __syncthreads();}
//	if (blockDim.x >= 128) {if (threadIdx.x < 64) smem[threadIdx.x] += smem[threadIdx.x + 64]; __syncthreads();}
//	if(threadIdx.x < 32)
//	{
//		if (blockDim.x >= 64) smem[threadIdx.x] += smem[threadIdx.x + 32];
//		if (blockDim.x >= 32) smem[threadIdx.x] += smem[threadIdx.x + 16];
//		if (blockDim.x >= 16) smem[threadIdx.x] += smem[threadIdx.x + 8];
//		if (blockDim.x >= 8) smem[threadIdx.x] += smem[threadIdx.x + 4];
//		if (blockDim.x >= 4) smem[threadIdx.x] += smem[threadIdx.x + 2];
//		if (blockDim.x >= 2) smem[threadIdx.x] += smem[threadIdx.x + 1];
//	}

    int idx=blockDim.x>>1;
    while(idx)
    {
        if(threadIdx.x<idx)
            smem[threadIdx.x]+=smem[threadIdx.x+idx];

        __syncthreads();
        idx>>=1;
    }
    __syncthreads();

	//Grid sum: use atomic operatin
	if(threadIdx.x == 0) atomicAdd(total, smem[0]);
}


/* meta_data.cuh makes sure one warp can do CTA scan*/
template<typename data_t, typename index_t>
__forceinline__ __device__ void _grid_scan(
		const index_t thd_id_inwarp,
		const index_t warp_id_inblk,
		const index_t warp_count_inblk,
		const data_t input, 
		data_t &output,
		data_t *smem,
		volatile data_t *total_sz
){
	data_t value = input;
	data_t my_cta_off = 0;

	//Warp scan: Kogge stone inclusive scan
	warp_scan_inc<data_t, index_t>(value,thd_id_inwarp);
	
	//CTA scan: use warp scan to do CTA scan
	if(thd_id_inwarp==31) smem[warp_id_inblk]=value;
	__syncthreads();
	if(warp_id_inblk==0)
	{
		data_t value2 = smem[thd_id_inwarp] * 
			(thd_id_inwarp < warp_count_inblk);
		
		warp_scan_exc<data_t, index_t>(value2, thd_id_inwarp);
		smem[thd_id_inwarp] = value2;
	}
	__syncthreads();
	
	//Grid scan: use atomic operatin to do grid scan
	//last thread in the block knows the sum of its block
	if(threadIdx.x == blockDim.x - 1)
	{
		int my_sum = smem[warp_id_inblk] + value;
		my_cta_off = atomicAdd((int*)total_sz, my_sum);  
		smem[warp_count_inblk] = my_cta_off;
	}
	__syncthreads();
	my_cta_off = smem[warp_count_inblk];

	output = value - input + smem[warp_id_inblk] + my_cta_off;
	__syncthreads();
}

/* meta_data.cuh makes sure one warp can do CTA scan*/
template<typename data_t, typename index_t>
__forceinline__ __device__ void _grid_scan_agg(
		const index_t thd_id_inwarp,
		const index_t warp_id_inblk,
		const index_t warp_count_inblk,
		const data_t input_sml, 
		const data_t input_mid, 
		const data_t input_lrg, 
		data_t &output_sml,
		data_t &output_mid,
		data_t &output_lrg,
		volatile data_t *total_sz_sml,
		volatile data_t *total_sz_mid,
		volatile data_t *total_sz_lrg
){
	__shared__ data_t smem_sml[32];
	__shared__ data_t smem_mid[32];
	__shared__ data_t smem_lrg[32];

	data_t value_sml = input_sml;
	data_t value_mid = input_mid;
	data_t value_lrg = input_lrg;
	
	data_t my_cta_off_sml;
	data_t my_cta_off_mid;
	data_t my_cta_off_lrg;

	//Warp scan: Kogge stone inclusive scan
	warp_scan_inc<data_t,index_t>(value_sml,thd_id_inwarp);
	warp_scan_inc<data_t,index_t>(value_mid,thd_id_inwarp);
	warp_scan_inc<data_t,index_t>(value_lrg,thd_id_inwarp);
	if(thd_id_inwarp == 31) 
	{
		smem_sml[warp_id_inblk] = value_sml;
		smem_mid[warp_id_inblk] = value_mid;
		smem_lrg[warp_id_inblk] = value_lrg;
	}
	__syncthreads();

	data_t my_warp_sum_sml = smem_sml[warp_id_inblk];
	data_t my_warp_sum_mid = smem_mid[warp_id_inblk];
	data_t my_warp_sum_lrg = smem_lrg[warp_id_inblk];
	
	__syncthreads();
	
	//CTA scan: use warp scan to do CTA scan
	data_t value2; 
	switch (warp_id_inblk)
	{
		case 0:
			value2 = smem_sml[thd_id_inwarp] *
				(thd_id_inwarp < warp_count_inblk);
			warp_scan_inc<data_t, index_t>
				(value2, thd_id_inwarp);
			smem_sml[thd_id_inwarp] = value2;
			break;

		case 1:
			value2 = smem_mid[thd_id_inwarp] *
				(thd_id_inwarp < warp_count_inblk);
			warp_scan_inc<data_t, index_t>
				(value2, thd_id_inwarp);
			smem_mid[thd_id_inwarp] = value2;
			break;	

		case 2:
			value2 = smem_lrg[thd_id_inwarp] *
				(thd_id_inwarp < warp_count_inblk);
			warp_scan_inc<data_t, index_t>
				(value2, thd_id_inwarp);
			smem_lrg[thd_id_inwarp] = value2;
			break;

		default:
			break;

	}
	__syncthreads();
	
	//Grid scan: use atomic operatin to do grid scan
	int my_sum;
	switch (warp_id_inblk)
	{
		case 0:
			my_sum = smem_sml[warp_count_inblk - 1];
			if(!thd_id_inwarp)
			{
				my_cta_off_sml = atomicAdd((int *)
					total_sz_sml, my_sum);
				smem_sml[warp_count_inblk] = 
					my_cta_off_sml;
			}
			break;

		case 1:
			my_sum = smem_mid[warp_count_inblk - 1];
			if(!thd_id_inwarp)
			{
				my_cta_off_mid = atomicAdd((int *)
					total_sz_mid, my_sum);
				smem_mid[warp_count_inblk] = 
					my_cta_off_mid;
			}
			break;

		case 2:
			my_sum = smem_lrg[warp_count_inblk - 1];
			if(!thd_id_inwarp)
			{
				my_cta_off_lrg = atomicAdd((int *)
					total_sz_lrg, my_sum);
				smem_lrg[warp_count_inblk] = 
					my_cta_off_lrg;
			}
			break;

		default:
			break;

	}
	__syncthreads();
	my_cta_off_sml = smem_sml[warp_count_inblk];
	my_cta_off_mid = smem_mid[warp_count_inblk];
	my_cta_off_lrg = smem_lrg[warp_count_inblk];

	output_sml = value_sml - input_sml +
		smem_sml[warp_id_inblk] - my_warp_sum_sml 
		+ my_cta_off_sml;

	output_mid = value_mid - input_mid +
		smem_mid[warp_id_inblk] - my_warp_sum_mid 
		+ my_cta_off_mid;

	output_lrg = value_lrg - input_lrg +
		smem_lrg[warp_id_inblk] - my_warp_sum_lrg 
		+ my_cta_off_lrg;

	__syncthreads();
}

template<typename data_t, typename index_t>
__global__ void grid_scan(meta_data mdata){
	__shared__ vertex_t smem[32];
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t thd_id_inwarp = threadIdx.x & 31;
	const index_t warp_id_inblk = threadIdx.x >> 5;
    const index_t warp_count_inblk = blockDim.x >> 5;
    
    data_t input = mdata.cat_thd_count_mid[TID];
    data_t output;
    
    _grid_scan(thd_id_inwarp, warp_id_inblk,
            warp_count_inblk, input, output, smem, 
            mdata.worklist_sz_mid);

    mdata.cat_thd_off_mid[TID] = output;
}

/*Pre scan*/
template <typename data_t, typename index_t>
__global__ void __prefix_pre_scan(	
		data_t 	*scan_in_d, 
		data_t 	*scan_out_d,
		data_t 	*blk_sum,
		index_t num_dat,
		const index_t THD_NUM)
{
	const data_t tile_sz= THD_NUM<<1;
	const index_t lane	= threadIdx.x<<1;
	index_t tid			= threadIdx.x+blockIdx.x*blockDim.x;
	index_t offset			= 1;
	index_t	tid_strip		= tid<<1;
	const index_t padding	= CONFLICT_FREE_OFFSET(tile_sz-1);

	//conflict free
	const index_t off_a		= CONFLICT_FREE_OFFSET(lane);
	const index_t off_b		= CONFLICT_FREE_OFFSET(lane+1);
	const index_t GRNTY = blockDim.x*gridDim.x;

	//prefetching danger
	extern __shared__ data_t s_mem[];

	while(tid_strip < num_dat)
	{
		s_mem[lane+off_a] 		= scan_in_d[tid_strip];
		s_mem[lane + 1 + off_b]	= scan_in_d[tid_strip + 1];
		__syncthreads();

		//Get each block sum, aside did the local prefix-sum
		for(index_t j=THD_NUM;j>0;j>>=1)
		{   
			if(threadIdx.x < j)
			{
				index_t ai	= offset*lane +offset - 1;
				index_t bi	= ai + offset;
				ai		   += CONFLICT_FREE_OFFSET(ai);
				bi		   += CONFLICT_FREE_OFFSET(bi);
				s_mem[bi]  += s_mem[ai];
			}
			offset <<=1;
			__syncthreads();
		}   
		__syncthreads();

		//write the block sum
		if(threadIdx.x == 0)
		{
			blk_sum[tid/THD_NUM]	= 
				s_mem[tile_sz-1+padding];
			s_mem[tile_sz-1+padding]		= 0;
		}
		__syncthreads();

		//Get the inter-block, prefix sum
		for(index_t j=1; j < tile_sz; j <<=1)
		{
			offset	>>=	1;
			if(threadIdx.x < j)
			{
				index_t ai	= lane*offset + offset - 1;
				index_t bi	= ai + offset;
				ai		   += CONFLICT_FREE_OFFSET(ai);
				bi		   += CONFLICT_FREE_OFFSET(bi);
				index_t t 	= s_mem[ai];
				s_mem[ai]	= s_mem[bi];
				s_mem[bi]  += t;
			}
			__syncthreads();
		}
		__syncthreads();
		scan_out_d[tid_strip]=s_mem[lane+off_a];
		scan_out_d[tid_strip+1]=s_mem[lane+1+off_b];

		tid += GRNTY;
		tid_strip = tid<<1;
	}
}

/*post scan*/
	template<typename data_t, typename index_t>
__global__ void __prefix_post_scan(
		index_t	*ex_q_sz_d,
		data_t 	*scan_out_d,
		data_t 	*blk_sum,
		index_t	num_dat,
		index_t num_blk,
		const index_t THD_NUM
		)
{
	extern __shared__ data_t s_mem[];
	const data_t tile_sz	= THD_NUM<<1;
	const index_t lane		= threadIdx.x<<1;
	index_t tid				= threadIdx.x+blockIdx.x*blockDim.x;
	index_t offset			= 1;
	const index_t padding	= CONFLICT_FREE_OFFSET(tile_sz - 1);
	const index_t off_a		= CONFLICT_FREE_OFFSET(lane);
	const index_t off_b		= CONFLICT_FREE_OFFSET(lane+1);

	/*Grid Scan*/
	if(lane<num_blk)
	{
		s_mem[lane+off_a]=blk_sum[lane];
	}else{
		s_mem[lane+off_a]=0;
	}

	if(lane+1<num_blk)
	{
		s_mem[lane+1+off_b]=blk_sum[lane+1];
	}else{
		s_mem[lane+1+off_b]=0;
	}
	__syncthreads();

	//up sweep
	for(index_t j = THD_NUM;j > 0;j>>=1)
	{   
		if(threadIdx.x < j)
		{
			index_t ai	= offset*lane +offset - 1;
			index_t bi	= ai + offset;

			ai		   += CONFLICT_FREE_OFFSET(ai);
			bi		   += CONFLICT_FREE_OFFSET(bi);
			s_mem[bi] += s_mem[ai];
		}
		offset <<=1;
		__syncthreads();
	}   
	__syncthreads();

	//write the block sum
	if(!threadIdx.x)
	{
		if(!blockIdx.x)
			ex_q_sz_d[0]=s_mem[tile_sz-1+padding];
		s_mem[tile_sz-1+padding]= 0;
	}
	__syncthreads();

	//down sweep
	for(index_t j=1; j < tile_sz; j <<=1)
	{
		offset	>>=	1;
		if(threadIdx.x < j)
		{
			index_t ai	= lane*offset + offset - 1;
			index_t bi	= ai + offset;
			ai		   += CONFLICT_FREE_OFFSET(ai);
			bi		   += CONFLICT_FREE_OFFSET(bi);
			index_t t 	= s_mem[ai];
			s_mem[ai]	= s_mem[bi];
			s_mem[bi]  += t;
		}
		__syncthreads();
	}
	__syncthreads();

	tid 			<<= 1;
	index_t blk_idx_orig= blockIdx.x;
	index_t blk_idx		= blk_idx_orig + 
		CONFLICT_FREE_OFFSET(blk_idx_orig);
	const index_t P_DATA= gridDim.x*blockDim.x*2;

	while (tid<num_dat)
	{
		scan_out_d[tid]=scan_out_d[tid]+
			s_mem[blk_idx];	
		scan_out_d[tid + 1]=scan_out_d[tid+1]+ 
			s_mem[blk_idx];
		tid			+= P_DATA;
		blk_idx_orig+= gridDim.x;
		blk_idx		= blk_idx_orig +
			CONFLICT_FREE_OFFSET(blk_idx_orig);
	}
}

/*Inspection Scan*/
	template<typename data_t, typename index_t>
__host__ void prefix_scan(
		data_t 			*scan_in_d,
		data_t 			*scan_out_d,
		data_t			*blk_sum,
		const index_t	num_dat,
		const index_t	BLK_NUM,
		const index_t	THD_NUM,
		index_t			*ex_q_sz_d,
		cudaStream_t 	&stream)
{
	/*data_t *blk_sum;*/
	const size_t sz = sizeof(data_t);
	const index_t num_blk=THD_NUM<<1;
	const index_t padding=CONFLICT_FREE_OFFSET((THD_NUM<<1)-1);
	//cudaMalloc((void **)&blk_sum,sz*num_blk);
	if(num_blk>THD_NUM*BLK_NUM)
	{
		std::cout<<"scan temp is out-boundary\n";
		exit(-2);
	}

	__prefix_pre_scan<data_t, index_t>
		<<<BLK_NUM, THD_NUM, (padding+(THD_NUM<<1))*sz, stream>>>
		(
		 scan_in_d,
		 scan_out_d,
		 blk_sum,
		 num_dat,
		 THD_NUM
		);

//	cudaDeviceSynchronize();
	__prefix_post_scan<data_t, index_t>
		<<<BLK_NUM, THD_NUM, (padding+(THD_NUM<<1))*sz, stream>>>
		(
		 ex_q_sz_d,
		 scan_out_d,
		 blk_sum,
		 num_dat,
		 num_blk,
		 THD_NUM
		);	
}

#endif
