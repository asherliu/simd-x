#ifndef __REDUCER__
#define __REDUCER__

#include "util.h"
#include "header.h"
#include "prefix_scan.cuh"
#include "meta_data.cuh"
#include "gpu_graph.cuh"
#include <limits.h>
#include <assert.h>

/*User provided virtual function*/
typedef bool (*cb_reducer)
	(vertex_t, feature_t,vertex_t*,index_t*,
	 feature_t*, feature_t *);

/* Gather data from bin to global worklist: thread-strided manner*/
//template<typename vertex_t, typename index_t>
class reducer
{
	public:
		//variables
		vertex_t *adj_list;
		index_t *beg_pos;
		index_t vert_count;
		feature_t *vert_status;
		feature_t *vert_status_prev;
		vertex_t *worklist_sml;
		vertex_t *worklist_mid;
		vertex_t *worklist_lrg;
		volatile vertex_t *worklist_sz_sml;
		volatile vertex_t *worklist_sz_mid;
		volatile vertex_t *worklist_sz_lrg;

	    index_t *cat_thd_count_sml;
		index_t *cat_thd_count_mid;
        index_t *cat_thd_count_lrg;
		
		cb_reducer vert_selector_push;
		cb_reducer vert_selector_pull;

	public:
		//constructor
		reducer(gpu_graph ggraph, 
				meta_data mdata,
				cb_reducer user_reducer_push,
				cb_reducer user_reducer_pull)
		{
			adj_list = ggraph.adj_list;
			beg_pos = ggraph.beg_pos;
			vert_count = ggraph.vert_count;
			
			vert_status = mdata.vert_status;
			vert_status_prev = mdata.vert_status_prev;
			worklist_sml = mdata.worklist_sml;
			worklist_mid = mdata.worklist_mid;
			worklist_lrg = mdata.worklist_lrg;
			
			worklist_sz_sml = mdata.worklist_sz_sml;
			worklist_sz_mid = mdata.worklist_sz_mid;
			worklist_sz_lrg = mdata.worklist_sz_lrg;
			
			vert_selector_push = user_reducer_push;
			vert_selector_pull = user_reducer_pull;
	    
            cat_thd_count_sml = mdata.cat_thd_count_sml;
		    cat_thd_count_mid = mdata.cat_thd_count_mid;
            cat_thd_count_lrg = mdata.cat_thd_count_lrg;
		}

	public:
		//functions
	/* Gather data from bin to global worklist: thread-strided manner*/
	__forceinline__ __device__ void 
		_thread_stride_gather(
		 vertex_t *worklist,
		 vertex_t *worklist_bin,
		 vertex_t my_front_count,
		 vertex_t output_off,
		 const index_t bin_off
		){
			//Thread strided global FQ generation
			//gather all frontiers into global mdata.worklist
			vertex_t dest_end = output_off + my_front_count;
			while(output_off < dest_end)
				worklist[output_off++] = 
					worklist_bin[(--my_front_count) + bin_off];
		}

	/* Gather data from bin to global worklist: warp-strided manner*/
	//template<typename vertex_t, typename index_t>
	__forceinline__ __device__ void 
		_warp_stride_gather(
		 vertex_t *worklist,
		 vertex_t *worklist_bin,
		 vertex_t my_front_count,
		 vertex_t output_off,
		 const index_t bin_off,
		 const index_t WOFF)
		{
			vertex_t warp_front_count;
			index_t warp_input_off, warp_output_off, warp_dest_end;

			//Warp Stride 	
			for(int i = 0; i < 32; i ++)
			{
				//Comm has problem.
				//if(__all(my_front_count * (i==WOFF)) == 0) continue;
				//
				//Quickly decide whether need to proceed on this thread
				warp_front_count = my_front_count;
				warp_front_count = __shfl(warp_front_count, i);
				if(warp_front_count == 0) continue;

				warp_output_off = output_off;
				warp_input_off = bin_off;
				warp_output_off = __shfl(warp_output_off, i);
				warp_input_off = __shfl(warp_input_off, i);
				warp_dest_end = warp_output_off + warp_front_count;
				warp_input_off += WOFF;
				warp_output_off += WOFF;

				while(warp_output_off < warp_dest_end)
				{
					worklist[warp_output_off] =
						worklist_bin[warp_input_off];
					warp_output_off += 32;
					warp_input_off += 32;
				}
			}
		}

	/* Coalesced scan status array to generate 
	 *non-sorted* frontier queue in push*/
	__forceinline__ __device__ void 
		_push_coalesced_scan_single_random_list(
			    vertex_t *smem,	
                const index_t TID,
				const index_t wid_in_blk,
				const index_t tid_in_wrp,
				const index_t wcount_in_blk,
				const index_t GRNTY,
				feature_t level)
		{
			vertex_t my_front_mid = 0;
			for(vertex_t my_beg = TID; my_beg < vert_count; my_beg += GRNTY)
			{
				if((*vert_selector_push)
					(my_beg, level, adj_list, beg_pos, 
					 vert_status, vert_status_prev))
				//if(vert_status[my_beg] <= K)
                {
					index_t degree = beg_pos[my_beg + 1] 
						- beg_pos[my_beg];
					if(degree == 0) continue;

					my_front_mid++;
				}
			}
			__syncthreads();
			vertex_t my_front_off_mid = 0;
			
			//For debugging
			//cat_thd_count_mid[TID] = my_front_mid;

			//prefix-scan
			_grid_scan<vertex_t, vertex_t>
				(tid_in_wrp,
                 wid_in_blk,
                 wcount_in_blk,
				 my_front_mid,
				 my_front_off_mid,
				 smem,
                 worklist_sz_mid);

			for(vertex_t my_beg = TID; my_beg < vert_count; my_beg += GRNTY)
			{
				if((*vert_selector_push)
					(my_beg, level, adj_list, beg_pos, 
					 vert_status, vert_status_prev))
			//	if(vert_status[my_beg] <= K)
				{
					index_t degree = beg_pos[my_beg + 1]
						- beg_pos[my_beg];
					if(degree == 0) continue;

					worklist_mid[my_front_off_mid++] = my_beg;
					
				}

                if(my_beg < vert_count)
                    //make sure already activated ones are turned off
                    if(vert_status_prev[my_beg] != vert_status[my_beg])
                        vert_status_prev[my_beg] = vert_status[my_beg];
			}
			__syncthreads();
		}

	/* Coalesced scan status array to generate 
	 *non-sorted* frontier queue in push*/
	__forceinline__ __device__ void 
		_push_coalesced_scan_random_list(
				const index_t TID,
				const index_t WIDL,
				const index_t WOFF,
				const index_t WCOUNT,
				const index_t GRNTY,
				feature_t level)
		{
			vertex_t my_front_sml = 0;
			vertex_t my_front_mid = 0;
			vertex_t my_front_lrg = 0;
			
			for(vertex_t my_beg = TID; my_beg < vert_count; my_beg += GRNTY)
			{
				if((*vert_selector_push)
					(my_beg, level, adj_list, beg_pos, 
					 vert_status, vert_status_prev))
				{
					index_t degree = beg_pos[my_beg + 1] 
						- beg_pos[my_beg];
					if(degree == 0) continue;

					if(degree < SML_MID) my_front_sml++;
					else if(degree > MID_LRG) my_front_lrg++;
					else my_front_mid++;
				}
			}
			__syncthreads();
			vertex_t my_front_off_sml = 0;
			vertex_t my_front_off_mid = 0;
			vertex_t my_front_off_lrg = 0;
			
			//For debugging
			cat_thd_count_sml[TID] = my_front_sml;
			cat_thd_count_mid[TID] = my_front_mid;
			cat_thd_count_lrg[TID] = my_front_lrg;

			//prefix-scan
			assert(WCOUNT >= 3);
			_grid_scan_agg<vertex_t, vertex_t>
				(WOFF, WIDL, WCOUNT,
				 my_front_sml,
				 my_front_mid,
				 my_front_lrg,
				 my_front_off_sml,
				 my_front_off_mid,
				 my_front_off_lrg,
				 worklist_sz_sml,
				 worklist_sz_mid,
				 worklist_sz_lrg);

			for(vertex_t my_beg = TID; my_beg < vert_count; my_beg += GRNTY)
			{
				if((*vert_selector_push)
					(my_beg, level, adj_list, beg_pos, 
					 vert_status, vert_status_prev))
				{
					index_t degree = beg_pos[my_beg + 1]
						- beg_pos[my_beg];
					if(degree == 0) continue;

					if(degree < SML_MID)
						worklist_sml[my_front_off_sml++] = my_beg;
					else if(degree > MID_LRG) 
						worklist_lrg[my_front_off_lrg++] = my_beg;
					else worklist_mid[my_front_off_mid++] = my_beg;
					
				}
                
                if(my_beg < vert_count)
                    //make sure already activated ones are turned off
                    if(vert_status_prev[my_beg] != vert_status[my_beg])
                        vert_status_prev[my_beg] = vert_status[my_beg];
			}
			__syncthreads();
		}
	
    
    /* Coalesced scan status array to 
	   generate *sorted* frontier queue */
	__forceinline__ __device__ void 
		_pull_coalesced_scan_sorted_list(
				const index_t TID,
				const index_t WIDL,
				const index_t WOFF,
				const index_t WCOUNT,
				const index_t GRNTY,
				feature_t level)
		{
			vertex_t my_beg, my_end;
			vertex_t my_beg_const;
			unsigned int flags;

			//figure out my task range
			//-assign by unit (32 indices deemed as one unit)
			index_t task_count = vert_count/GRNTY;
			index_t remainder = vert_count - task_count * GRNTY;

			//figure out each thread's task offsets
			if(TID < remainder)
			{
				task_count++;
				my_beg = task_count * TID;
				my_end = my_beg + task_count;
			}
			else
			{
				my_beg = task_count * TID + remainder;
				my_end = my_beg + task_count;
			}

			//ensure no out-of-boundary error
			assert(my_end <= vert_count);

			my_beg_const = my_beg;
			vertex_t my_front_sml = 0;
			vertex_t my_front_mid = 0;
			vertex_t my_front_lrg = 0;
			
			//For debugging
			//	while(my_beg < my_end)
			//	{
			//		vertex_t vert_id = my_beg;
			//		if(vert_status[vert_id] == INFTY)
			//		{
			//			index_t degree = beg_pos[vert_id + 1] 
			//				- beg_pos[vert_id];
			//			if(degree == 0)
			//			{
			//				my_beg ++;
			//				continue;
			//			}
			//			if(degree < SML_MID) my_front_sml++;
			//			else if(degree > MID_LRG) my_front_lrg++;
			//			else my_front_mid++;
			//		}
			//		my_beg ++;
			//	}
			//	__syncthreads();

			//ballot for each threads of my warp
			while(__syncthreads_or(my_beg < my_end))
			{
				flags = 0;
				//load results to shared memory, warp strided
				for(unsigned i = 0; i < 32; i ++)
				{
					//the whole warp is working on laneid=i's range
					int curr_beg;
					int curr_end;
					if(i == WOFF) 
					{
						curr_beg = my_beg;
					}   curr_end = my_end;

					curr_beg = __shfl(curr_beg, i);
					curr_end = __shfl(curr_end, i);

					unsigned int vote = 0;
					int predicate = 0;

					//absent thread will be 0.
					//according to CUDA programming
					//-not true!!!!
					if(curr_beg + WOFF < curr_end)
						predicate = (*vert_selector_pull)
							(curr_beg + WOFF, level, adj_list, beg_pos, 
							 vert_status, vert_status_prev)==true;

					vote = __ballot(predicate);
					if(WOFF == i) flags = vote;
				}

				//check loaded results, thread strided
				for(unsigned i = 0; i < 32; i ++)
				{
					if(flags & (((unsigned int)1) << i))
					{
						index_t vert_id = my_beg + i;
						index_t degree = beg_pos[vert_id + 1] 
							- beg_pos[vert_id];

						if(degree == 0) continue;
						if(degree < SML_MID) my_front_sml++;
						else if(degree > MID_LRG) my_front_lrg++;
						else my_front_mid++;
					}
				}
				my_beg += 32;
			}

			vertex_t my_front_off_sml = 0;
			vertex_t my_front_off_mid = 0;
			vertex_t my_front_off_lrg = 0;
			
			//For debugging
			//mdata.cat_thd_count_sml[TID] = my_front_sml;
			//mdata.cat_thd_count_mid[TID] = my_front_mid;
			//mdata.cat_thd_count_lrg[TID] = my_front_lrg;

			//prefix-scan
			assert(WCOUNT >= 3);
			_grid_scan_agg<vertex_t, vertex_t>
				(WOFF, WIDL, WCOUNT,
				 my_front_sml,
				 my_front_mid,
				 my_front_lrg,
				 my_front_off_sml,
				 my_front_off_mid,
				 my_front_off_lrg,
				 worklist_sz_sml,
				 worklist_sz_mid,
				 worklist_sz_lrg);

			//ballot for each threads of my warp
			my_beg = my_beg_const;

			//For debugging
			//while(my_beg < my_end)
			//{
			//	vertex_t vert_id = my_beg;
			//	if(vert_status[vert_id] == INFTY)
			//	{
			//		index_t degree = beg_pos[vert_id + 1]
			//			- beg_pos[vert_id];
			//		if(degree == 0)
			//		{
			//			my_beg ++;
			//			continue;
			//		}
			//		if(degree < SML_MID)
			//			worklist_sml[my_front_off_sml++] = vert_id;
			//		else if(degree > MID_LRG) 
			//			worklist_lrg[my_front_off_lrg++] = vert_id;
			//		else worklist_mid[my_front_off_mid++] = vert_id;
			//	}
			//	my_beg ++;
			//}
			//__syncthreads();

			while(__syncthreads_or(my_beg < my_end))
			{
				flags = 0;
				//load results to shared memory, warp strided
				for(int i = 0; i < 32; i ++)
				{
					//the whole warp is working on laneid=i's range
					index_t curr_beg;
					index_t curr_end;
					if(i == WOFF) 
					{
						curr_beg = my_beg;
					}   curr_end = my_end;

					curr_beg = __shfl(my_beg, i);
					curr_end = __shfl(my_end, i);

					//in case this whole warp just goes out of boundary
					//cannot terminate because needs to join syncthreads
					unsigned int vote = 0;
					int predicate = 0;

					//absent thread will be 0.
					//according to CUDA programming
					//-not true!!!!
					if(curr_beg + WOFF < curr_end)
						predicate = (*vert_selector_pull)
							(curr_beg + WOFF, level, adj_list, beg_pos, 
							 vert_status, vert_status_prev)==true;

					vote = __ballot(predicate);
					if(WOFF == i) flags = vote;
				}

				//check loaded results, thread strided
				for(int i = 0; i < 32; i ++)
				{
					index_t vert_id = my_beg + i;
					if(flags&((unsigned int)1<<i))
					{
						index_t degree = beg_pos[vert_id + 1]
							- beg_pos[vert_id];

						if(degree == 0) continue;

						if(degree < SML_MID)
							worklist_sml[my_front_off_sml++] = vert_id;
						else if(degree > MID_LRG) 
							worklist_lrg[my_front_off_lrg++] = vert_id;
						else worklist_mid[my_front_off_mid++] = vert_id;
					
					}
				    
                    if(vert_id < vert_count)
                        if(vert_status_prev[vert_id] != vert_status[vert_id])
                            //make sure already activated ones are turned off
                            vert_status_prev[vert_id] = vert_status[vert_id];
					
				}
				my_beg += 32;
			}
		}

	/* strided scan status array to 
	   generate *sorted* frontier queue */
	__forceinline__ __device__ void 
		_pull_strided_scan_sorted_list(
				const index_t TID,
				const index_t WIDL,
				const index_t WOFF,
				const index_t WCOUNT,
				const index_t GRNTY,
				feature_t level)
		{
			vertex_t my_beg, my_end;
			vertex_t my_beg_const;
			unsigned int flags;

			//figure out my task range
			//-assign by unit (32 indices deemed as one unit)
			index_t task_count = vert_count/GRNTY;
			index_t remainder = vert_count - task_count * GRNTY;

			//figure out each thread's task offsets
			if(TID < remainder)
			{
				task_count++;
				my_beg = task_count * TID;
				my_end = my_beg + task_count;
			}
			else
			{
				my_beg = task_count * TID + remainder;
				my_end = my_beg + task_count;
			}

			//ensure no out-of-boundary error
			assert(my_end <= vert_count);

			my_beg_const = my_beg;
			vertex_t my_front_sml = 0;
			vertex_t my_front_mid = 0;
			vertex_t my_front_lrg = 0;
			
			//For debugging
			//	while(my_beg < my_end)
			//	{
			//		vertex_t vert_id = my_beg;
			//		if(vert_status[vert_id] == INFTY)
			//		{
			//			index_t degree = beg_pos[vert_id + 1] 
			//				- beg_pos[vert_id];
			//			if(degree == 0)
			//			{
			//				my_beg ++;
			//				continue;
			//			}
			//			if(degree < SML_MID) my_front_sml++;
			//			else if(degree > MID_LRG) my_front_lrg++;
			//			else my_front_mid++;
			//		}
			//		my_beg ++;
			//	}
			//	__syncthreads();

			//every thread works on its own stride
            for(; my_beg < my_end; my_beg ++)
            {
                if((*vert_selector_pull)
                        (my_beg, level, adj_list, beg_pos, 
                         vert_status, vert_status_prev)==true)
                {
                    index_t degree = beg_pos[my_beg + 1] 
                        - beg_pos[my_beg];

                    if(degree == 0) continue;
                    if(degree < SML_MID) my_front_sml++;
                    else if(degree > MID_LRG) my_front_lrg++;
                    else my_front_mid++;
                }
            }

			vertex_t my_front_off_sml = 0;
			vertex_t my_front_off_mid = 0;
			vertex_t my_front_off_lrg = 0;
			
			//For debugging
			//mdata.cat_thd_count_sml[TID] = my_front_sml;
			//mdata.cat_thd_count_mid[TID] = my_front_mid;
			//mdata.cat_thd_count_lrg[TID] = my_front_lrg;

			//prefix-scan
			assert(WCOUNT >= 3);
			_grid_scan_agg<vertex_t, vertex_t>
				(WOFF, WIDL, WCOUNT,
				 my_front_sml,
				 my_front_mid,
				 my_front_lrg,
				 my_front_off_sml,
				 my_front_off_mid,
				 my_front_off_lrg,
				 worklist_sz_sml,
				 worklist_sz_mid,
				 worklist_sz_lrg);

			//ballot for each threads of my warp
			my_beg = my_beg_const;

			//For debugging
			//while(my_beg < my_end)
			//{
			//	vertex_t vert_id = my_beg;
			//	if(vert_status[vert_id] == INFTY)
			//	{
			//		index_t degree = beg_pos[vert_id + 1]
			//			- beg_pos[vert_id];
			//		if(degree == 0)
			//		{
			//			my_beg ++;
			//			continue;
			//		}
			//		if(degree < SML_MID)
			//			worklist_sml[my_front_off_sml++] = vert_id;
			//		else if(degree > MID_LRG) 
			//			worklist_lrg[my_front_off_lrg++] = vert_id;
			//		else worklist_mid[my_front_off_mid++] = vert_id;
			//	}
			//	my_beg ++;
			//}
			//__syncthreads();
			
            //every thread works on its own stride
            for(; my_beg < my_end; my_beg ++)
            {
                if((*vert_selector_pull)
                        (my_beg, level, adj_list, beg_pos, 
                         vert_status, vert_status_prev)==true)
                {
                    index_t degree = beg_pos[my_beg + 1] 
                        - beg_pos[my_beg];

                    if(degree == 0) continue;
                    if(degree < SML_MID) worklist_sml[my_front_off_sml++] = my_beg;
                    else if(degree > MID_LRG) worklist_lrg[my_front_off_lrg++] = my_beg;
                    else worklist_mid[my_front_off_mid++] = my_beg;
					
                }
                
                if(my_beg < vert_count)
                    if(vert_status_prev[my_beg] != vert_status[my_beg])
                        //make sure already activated ones are turned off
                        vert_status_prev[my_beg] = vert_status[my_beg];
            }
		}
};

#endif
