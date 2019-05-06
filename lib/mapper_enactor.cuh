#ifndef _ENACTOR_H_
#define _ENACTOR_H_
#include "gpu_graph.cuh"
#include "meta_data.cuh"
#include "reducer_enactor.cuh"

#include "util.h"
#include "header.h"
#include "prefix_scan.cuh"
#include <limits.h>
#include "barrier.cuh"

//Push model: one kernel for multiple iterations
__global__ void
merge_push_kernel(
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier
){
	__shared__ vertex_t smem[32];
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;
	const index_t wid_in_blk = threadIdx.x >> 5;
	const index_t wid_in_grd = TID >> 5;
	const index_t wcount_in_blk = blockDim.x >> 5;
	const index_t WGRNTY = GRNTY >> 5;
	const index_t BIN_OFF = TID * BIN_SZ;

	feature_t level_thd = level[0];
	vertex_t output_off;
	
	//vertex_t mdata.mdata.worklist_bin_reg[16]; 
	//Not a big difference comparing to 
	//directly store frontiers in global mem
#ifdef __AGG_SUB__
    worklist_gather._push_coalesced_scan_single_random_list
        (smem,TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level_thd);
#endif
	vertex_t mid_queue = mdata.worklist_sz_mid[0]; 

//	if(!TID) printf("outlevel-%d: %d\n", (int)level_thd, mid_queue);
	while(true)
	{
	  /* CANNOT immediately change worklist_sz_sml[0] to 0
	   * global_barrier could sync threads, not memory updates
	   * ****if(!TID) mdata.worklist_sz_sml[0] = 0;*******
	   * should be far away after
	   * ***if((wqueue = mdata.worklist_sz_sml[0]) == 0) break;****
	   */
		global_barrier.sync_grid_opt();
		
		if(!TID) 
        {    
            mdata.worklist_sz_mid[0] = 0;
            //mdata.worklist_sz_sml[0] = 0;
        }

		//global_barrier.sync_grid_opt();
		
        vertex_t my_front_count = 0;
		//compute on the graph 
		//and generate frontiers immediately
		
        index_t appr_work = 0;
		compute_mapper.mapper_bin_push_online_alone(
                appr_work,
				my_front_count,
				mdata.worklist_bin,
				mid_queue,
				mdata.worklist_mid,
				//wid_in_grd,/*group id*/
				//32,/*group size*/
				//WGRNTY,/*group count*/
				//tid_in_wrp,/*thread off intra group*/
				TID,/*group id*/
				1,/*group size*/
				GRNTY,/*group count*/
				0,/*thread off intra group*/
				level_thd, 
				BIN_OFF);
        //assert(mdata.worklist_sz_sml[0] != -1);

		//prefix_scan
		_grid_scan<vertex_t, vertex_t>
			(tid_in_wrp, 
			 wid_in_blk, 
			 wcount_in_blk, 
			 my_front_count, 
			 output_off, 
			 smem,
			 mdata.worklist_sz_mid);

		////check if finished
		//break;
		global_barrier.sync_grid_opt();
		if((mid_queue = mdata.worklist_sz_mid[0]) == 0 )break;//||
		//		mdata.worklist_sz_mid[0]*ggraph.avg_degree > (GRNTY<<2)) break;
#ifdef ENABLE_MONITORING
        if(!TID) printf("level-%d: %d\n", (int)level_thd, mid_queue);
		//global_barrier.sync_grid_opt();
#endif

		//compact all thread bins in frontier queue	

		worklist_gather._thread_stride_gather
			(mdata.worklist_mid, 
			 mdata.worklist_bin, 
			 my_front_count, 
			 output_off, 
			 BIN_OFF);

		level_thd ++;
	}
	if(!TID) level[0] = level_thd;
}



//Push model: one kernel for multiple iterations
__global__ void
hybrid_bin_scan_push_kernel(
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier
){
	__shared__ vertex_t smem[32];
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;
	const index_t wid_in_blk = threadIdx.x >> 5;
	const index_t wid_in_grd = TID >> 5;
	const index_t wcount_in_blk = blockDim.x >> 5;
	const index_t WGRNTY = GRNTY >> 5;
	const index_t BIN_OFF = TID * BIN_SZ;

	feature_t level_thd = level[0];
	vertex_t output_off;
	
	//vertex_t mdata.mdata.worklist_bin_reg[16]; 
	//Not a big difference comparing to 
	//directly store frontiers in global mem
    if(!TID) mdata.worklist_sz_mid[0] = 0;
	
    global_barrier.sync_grid_opt();
    worklist_gather._push_coalesced_scan_single_random_list
        (smem,TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level_thd);
	vertex_t mid_queue = mdata.worklist_sz_mid[0]; 
    
#ifdef ENABLE_MONITORING
    if(!TID) printf("Entering-hybrid-bin-front-count: %d\n",mid_queue);
#endif

	while(true)
	{
	  /* CANNOT immediately change worklist_sz_sml[0] to 0
	   * global_barrier could sync threads, not memory updates
	   * ****if(!TID) mdata.worklist_sz_sml[0] = 0;*******
	   * should be far away after
	   * ***if((wqueue = mdata.worklist_sz_sml[0]) == 0) break;****
	   */
        mdata.future_work[0] = 0;
		global_barrier.sync_grid_opt();
		
		if(!TID){
            mdata.worklist_sz_mid[0] = 0;
            mdata.worklist_sz_sml[0] = 0;//indicate whether bin overflow
        }
		vertex_t my_front_count = 0;

		//compute on the graph 
		//and generate frontiers immediately
		global_barrier.sync_grid_opt();
#ifdef ENABLE_MONITORING
        if(!TID) printf("level-%d-frontier-count: %d\n", (int)level_thd,mid_queue);
#endif
	    index_t appr_work = 0;	
		
        //Online filter is included.
        //-Comment out recoder to disable online filter.
        compute_mapper.mapper_bin_push(
				appr_work,
                mdata.worklist_sz_sml,
                my_front_count,
				mdata.worklist_bin,
				mid_queue,
				mdata.worklist_mid,
                wid_in_grd,/*group id*/
                32,/*group size*/
                WGRNTY,/*group count*/
                tid_in_wrp,/*thread off intra group*/
				level_thd, 
				BIN_OFF);

        //global_barrier.sync_grid_opt();
        
        _grid_sum<vertex_t, index_t>(appr_work, mdata.future_work);
        global_barrier.sync_grid_opt();
        if(mdata.future_work[0] > ggraph.edge_count * SWITCH_TO)
        {
#ifdef ENABLE_MONITORING
            if (!TID) printf("----->>>Switch to pull model<<<-----------\n");
#endif
            break;
        }
        //worklist_gather._push_coalesced_scan_single_random_list
        //    (smem,TID, wid_in_blk, tid_in_wrp,wcount_in_blk,GRNTY,level_thd+1);
        
        if(mdata.worklist_sz_sml[0] == -1)//means overflow
        //if(true)// - Intentionally always overflow, for the purpose of test online filter overhead.
        {
#ifdef ENABLE_MONITORING
            if(!TID) printf("------->>>Switch to Ballot filtering<<<-------\n");
#endif

            worklist_gather._push_coalesced_scan_single_random_list
                (smem,TID, wid_in_blk, tid_in_wrp,wcount_in_blk,GRNTY,level_thd+1);
        }
        else
        {
#ifdef ENABLE_MONITORING
            if(!TID) printf("--->>>>>>Online-Filter<<<<<<-----------\n");
#endif
            //Attention, its likely frontier list size goes beyond vert_count
            _grid_scan<vertex_t, vertex_t>
                (tid_in_wrp, 
                 wid_in_blk, 
                 wcount_in_blk, 
                 my_front_count, 
                 output_off, 
                 smem,
                 mdata.worklist_sz_mid);

            //compact all thread bins in frontier queue	
            worklist_gather._thread_stride_gather
                (mdata.worklist_mid, 
                 mdata.worklist_bin, 
                 my_front_count, 
                 output_off, 
                 BIN_OFF);
        }	
            
        global_barrier.sync_grid_opt();
        if((mid_queue = mdata.worklist_sz_mid[0]) == 0)break;//||
#ifndef __VOTE__ 
        for(index_t i = TID; i<ggraph.vert_count;i+=GRNTY)
            mdata.vert_status_prev[i] = mdata.vert_status[i];
#endif
		level_thd ++;

        //if (level_thd == 20) break;
	}
	if(!TID) level[0] = level_thd;
}


__global__ void
merge_pull_kernel(
	feature_t terminate_level,
	feature_t *level_record,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier
){
	//__shared__ vertex_t smem[32];
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;
	const index_t wid_in_blk = threadIdx.x >> 5;
	const index_t wid_in_grd = TID >> 5;
	const index_t wcount_in_blk = blockDim.x >> 5;
	const index_t WGRNTY = GRNTY >> 5;
	
	feature_t level = 0;
	//vertex_t mdata.mdata.worklist_bin_reg[16]; 
	//Not a big difference comparing to 
	//directly store frontiers in global mem

#ifndef __VOTE__//NONE vote algorithms
    if(!TID)
    {
        mdata.worklist_sz_sml[0] = 0; 
        mdata.worklist_sz_mid[0] = 0; 
        mdata.worklist_sz_lrg[0] = 0;
    }
    global_barrier.sync_grid_opt();
    worklist_gather._pull_coalesced_scan_sorted_list
        (TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level+level_record[0]);
    global_barrier.sync_grid_opt();
#endif

    while(true)
	{
#ifdef __VOTE__ //Vote algorithms
        if(!TID)
        {
            mdata.worklist_sz_sml[0] = 0; 
            mdata.worklist_sz_mid[0] = 0; 
            mdata.worklist_sz_lrg[0] = 0;
        }
        global_barrier.sync_grid_opt();
        worklist_gather._pull_coalesced_scan_sorted_list
            (TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level+level_record[0]);
        global_barrier.sync_grid_opt();
#else
    
        for(index_t i = TID; i<ggraph.vert_count;i+=GRNTY)
            mdata.vert_status_prev[i] = mdata.vert_status[i];
        global_barrier.sync_grid_opt();
#endif

       //if (!TID) mdata.future_work[0] = 0;
		//global_barrier.sync_grid_opt();

        //_grid_sum<vertex_t, index_t>(mdata.cat_thd_count_sml[TID] + 
        //        mdata.cat_thd_count_mid[TID] + 
        //        mdata.cat_thd_count_lrg[TID], mdata.future_work);
        //if (!TID) printf("level-%d-update-%d, vert_count-%d, edge_count-%d\n",
        //        (int)(level+level_record[0]), mdata.future_work[0], ggraph.vert_count, ggraph.edge_count);
#ifdef ENABLE_MONITORING
        if (!TID) printf("level-%d-break after %d levels\n",(int)(level+level_record[0]), terminate_level-level);
#endif
        //if(mdata.future_work[0] < ggraph.vert_count * SWITCH_BACK) break;


		//Three pull mappers
        compute_mapper.cta_mapper_pull(
                mdata.cat_thd_count_lrg,
                mdata.worklist_sz_lrg[0], 
                mdata.worklist_lrg, 
                level_record[0]+level);
        compute_mapper.warp_mapper_pull(
                mdata.cat_thd_count_mid,
                mdata.worklist_sz_mid[0], 
                mdata.worklist_mid, 
                wid_in_grd, 
                tid_in_wrp, 
                WGRNTY,/*group count*/
                level_record[0]+level);
        compute_mapper.thd_mapper_pull(
                mdata.cat_thd_count_sml,
                mdata.worklist_sz_sml[0], 
                mdata.worklist_sml, 
                TID, 
                GRNTY, 
                level_record[0]+level);
//		global_barrier.sync_grid_opt();

		level ++;
        if(level == terminate_level) 
        {

#ifdef ENABLE_MONITORING
            if (!TID) printf("-------->>>Switch to push model<<<<<<-------\n");
#endif
            break;
        }
//		global_barrier.sync_grid_opt();
	}

    if (!TID) level_record[0] += level;
}


__global__ void
balanced_push_kernel(
	feature_t *level_record,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier
){
	//__shared__ vertex_t smem[32];
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;
	const index_t wid_in_blk = threadIdx.x >> 5;
	const index_t wid_in_grd = TID >> 5;
	const index_t wcount_in_blk = blockDim.x >> 5;
	const index_t WGRNTY = GRNTY >> 5;
	
	feature_t level = level_record[0];
	//vertex_t mdata.mdata.worklist_bin_reg[16]; 
	//Not a big difference comparing to 
    //directly store frontiers in global mem
    while(true)
    {
        /* CANNOT immediately change worklist_sz_sml[0] to 0
         * global_barrier could sync threads, not memory updates
         * ****if(!TID) mdata.worklist_sz_sml[0] = 0;*******
         * should be far away after
         * ***if((wqueue = mdata.worklist_sz_sml[0]) == 0) break;****
         */
        if(!TID)
        {
            mdata.worklist_sz_sml[0] = 0; 
            mdata.worklist_sz_mid[0] = 0; 
            mdata.worklist_sz_lrg[0] = 0;
        }
        global_barrier.sync_grid_opt();
        
        //assert(mdata.worklist_sz_sml[0] == 0);
        //assert(mdata.worklist_sz_mid[0] == 0);
        //assert(mdata.worklist_sz_lrg[0] == 0);
        
        worklist_gather._push_coalesced_scan_random_list
            (TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level);

        mdata.future_work[0] = 0;
        global_barrier.sync_grid_opt();
        
#ifdef ENABLE_MONITORING
        if (!TID) printf("level-%d-count: %d\n", (int)level,mdata.worklist_sz_sml[0] + 
                mdata.worklist_sz_mid[0] + 
                mdata.worklist_sz_lrg[0]);
#endif

        if(mdata.worklist_sz_sml[0] + 
                mdata.worklist_sz_mid[0] + 
                mdata.worklist_sz_lrg[0] == 0) break;

        //global_barrier.sync_grid_opt();

        //Three push mappers.
        compute_mapper.mapper_push(
                mdata.worklist_sz_lrg[0],
                mdata.worklist_lrg,
                mdata.cat_thd_count_lrg,
                blockIdx.x,/*group id*/
                blockDim.x,/*group size*/
                gridDim.x,/*group count*/
                threadIdx.x,/*thread off intra group*/
                level);

        compute_mapper.mapper_push(
                mdata.worklist_sz_mid[0],
                mdata.worklist_mid,
                mdata.cat_thd_count_mid,
                wid_in_grd,/*group id*/
                32,/*group size*/
                WGRNTY,/*group count*/
                tid_in_wrp,/*thread off intra group*/
                level);

        compute_mapper.mapper_push(
                mdata.worklist_sz_sml[0],
                mdata.worklist_sml,
                mdata.cat_thd_count_sml,
                TID,/*group id*/
                1,/*group size*/
                GRNTY,/*group count*/
                0,/*thread off intra group*/
                level);

  //      global_barrier.sync_grid_opt();
        
        _grid_sum<vertex_t, index_t>(mdata.cat_thd_count_sml[TID] + 
                mdata.cat_thd_count_mid[TID] + 
                mdata.cat_thd_count_lrg[TID], mdata.future_work);
        
        level ++;
        global_barrier.sync_grid_opt();
        if(mdata.future_work[0] > ggraph.edge_count * SWITCH_TO)
        {
#ifdef ENABLE_MONITORING
            if (!TID) printf("-------->>>Switch to pull model<<<<<<-------\n");
#endif
            break;
        }
#ifdef ENABLE_MONITORING
        if (!TID) printf("level-%d-futurework: %d\n", (int)level,mdata.future_work[0]); 
#endif        
        //if(level == 1) break;
    }
	
    if(!TID) level_record[0] = level;
}

__global__ void
push_pull_fused_kernel(
        int iter_limit,
        bool is_topdown_input,
        feature_t *level_record,
        gpu_graph ggraph,
        meta_data mdata,
        mapper compute_mapper,
        reducer worklist_gather,
        Barrier global_barrier
){
	//__shared__ vertex_t smem[32];
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;
	const index_t wid_in_blk = threadIdx.x >> 5;
	const index_t wid_in_grd = TID >> 5;
	const index_t wcount_in_blk = blockDim.x >> 5;
	const index_t WGRNTY = GRNTY >> 5;
	
	vertex_t bt_level_count;
	feature_t level = 0;
	bool is_topdown = is_topdown_input;
	//vertex_t mdata.mdata.worklist_bin_reg[16]; 
	//Not a big difference comparing to 
	//directly store frontiers in global mem
	while(true)
	{
	  /* CANNOT immediately change worklist_sz_sml[0] to 0
	   * global_barrier could sync threads, not memory updates
	   * ****if(!TID) mdata.worklist_sz_sml[0] = 0;*******
	   * should be far away after
	   * ***if((wqueue = mdata.worklist_sz_sml[0]) == 0) break;****
	   */
		if(TID == 0)  mdata.worklist_sz_sml[0] = 0;
		if(TID == 32) mdata.worklist_sz_mid[0] = 0;
		if(TID == 64) mdata.worklist_sz_lrg[0] = 0;
		
		global_barrier.sync_grid_opt();
		if(is_topdown)
		{
			worklist_gather._push_coalesced_scan_random_list
                (TID, wid_in_blk, tid_in_wrp, wcount_in_blk,GRNTY,level);
            
#ifdef ENABLE_MONITORING
            if (!TID)
                printf("topdown-level - %d, fcount: %d\n",(int)level,
                        mdata.worklist_sz_sml[0] 
                        + mdata.worklist_sz_mid[0] 
                        + mdata.worklist_sz_lrg[0]);
#endif

            mdata.future_work[0] = 0;
			global_barrier.sync_grid_opt();

			if(mdata.worklist_sz_sml[0] + 
					mdata.worklist_sz_mid[0] + 
					mdata.worklist_sz_lrg[0] == 0) break;
			
			//Three push mappers.
			compute_mapper.mapper_push(
					mdata.worklist_sz_lrg[0],
					mdata.worklist_lrg,
					mdata.cat_thd_count_lrg,
					blockIdx.x,/*group id*/
					blockDim.x,/*group size*/
					gridDim.x,/*group count*/
					threadIdx.x,/*thread off intra group*/
					level);
			
			compute_mapper.mapper_push(
					mdata.worklist_sz_mid[0],
					mdata.worklist_mid,
					mdata.cat_thd_count_mid,
					wid_in_grd,/*group id*/
					32,/*group size*/
					WGRNTY,/*group count*/
					tid_in_wrp,/*thread off intra group*/
					level);

			
			compute_mapper.mapper_push(
					mdata.worklist_sz_sml[0],
					mdata.worklist_sml,
					mdata.cat_thd_count_sml,
					TID,/*group id*/
					1,/*group size*/
					GRNTY,/*group count*/
					0,/*thread off intra group*/
					level);
			
		    _grid_sum<vertex_t, index_t>(mdata.cat_thd_count_sml[TID] + 
					mdata.cat_thd_count_mid[TID] + 
					mdata.cat_thd_count_lrg[TID], mdata.future_work);
			global_barrier.sync_grid_opt();
			if(mdata.future_work[0] > ggraph.edge_count * SWITCH_TO) 
			{
				is_topdown = false;
				bt_level_count = 0;
                if(!TID) printf("switch to bottom-up-%d\n",mdata.future_work[0]);
			}
            //if(!TID) printf("there\n");
		}
		else
		{
#ifdef ENABLE_MONITORING
            if(!TID) printf("here, iterLimit%d,level%d\n", iter_limit, level);
#endif
            worklist_gather._pull_coalesced_scan_sorted_list
                (TID, wid_in_blk, tid_in_wrp,wcount_in_blk,GRNTY, level);
			mdata.future_work[0] = 0;
			global_barrier.sync_grid_opt();
			
			//Three pull mappers
			compute_mapper.cta_mapper_pull(
					mdata.cat_thd_count_lrg,
					mdata.worklist_sz_lrg[0], 
					mdata.worklist_lrg, 
					level);
			compute_mapper.warp_mapper_pull(
					mdata.cat_thd_count_mid,
					mdata.worklist_sz_mid[0], 
					mdata.worklist_mid, 
					wid_in_grd, 
					tid_in_wrp, 
					WGRNTY,/*group count*/
					level);
			compute_mapper.thd_mapper_pull(
					mdata.cat_thd_count_sml,
					mdata.worklist_sz_sml[0], 
					mdata.worklist_sml, 
					TID, 
					GRNTY, 
					level);
			
			_grid_sum<vertex_t, index_t>(mdata.cat_thd_count_sml[TID] + 
					mdata.cat_thd_count_mid[TID] + 
					mdata.cat_thd_count_lrg[TID], mdata.future_work);
			global_barrier.sync_grid_opt();
#ifdef ENABLE_MONITORING
            if (!TID) printf("bottom-up: update-%d, vert_count-%d, edge_count-%d\n",
                    mdata.future_work[0], ggraph.vert_count, ggraph.edge_count);
#endif

            if(mdata.future_work[0] < ggraph.vert_count * SWITCH_BACK) 
			{
				is_topdown = true;
#ifdef ENABLE_MONITORING
                if(!TID) printf("switch to top-down\n");
#endif
			}
			//bt_level_count++;
			//if(bt_level_count == 3)
			//	is_topdown = true;
		}   
		    
		global_barrier.sync_grid_opt();
		level ++;
        
        if(level == iter_limit) break;
	}
	if(!TID) level_record[0] = level;
}

//__global__ void
//push_pull_fused_fully_expand_kernel(
//	feature_t *level_record,
//	gpu_graph ggraph,
//	meta_data mdata,
//	mapper compute_mapper,
//	reducer worklist_gather,
//	Barrier global_barrier
//){
//	//__shared__ vertex_t smem[32];
//	__shared__ feature_t smem[THDS_NUM];
//	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
//	const index_t GRNTY = blockDim.x * gridDim.x;
//	const index_t tid_in_wrp = threadIdx.x & 31;
//	const index_t wid_in_blk = threadIdx.x >> 5;
//	const index_t wid_in_grd = TID >> 5;
//	const index_t wcount_in_blk = blockDim.x >> 5;
//	const index_t WGRNTY = GRNTY >> 5;
////	const index_t BIN_OFF = TID * BIN_SZ;
//	
//	vertex_t fq_sz = 0;
//	feature_t level = 0;
//	//vertex_t mid_queue = mdata.worklist_sz_mid[0]; 
//	//vertex_t output_off;
//	bool is_topdown = true;
////	bool is_topdown = false;
//	weight_t weight;			
//	index_t appr_work = 0;
//
//	//vertex_t mdata.mdata.worklist_bin_reg[16]; 
//	//Not a big difference comparing to 
//	//directly store frontiers in global mem
//	while(true)
//	{
//	  /* CANNOT immediately change worklist_sz_sml[0] to 0
//	   * global_barrier could sync threads, not memory updates
//	   * ****if(!TID) mdata.worklist_sz_sml[0] = 0;*******
//	   * should be far away after
//	   * ***if((wqueue = mdata.worklist_sz_sml[0]) == 0) break;****
//	   */
//
//		if(is_topdown)
//		{
//			worklist_gather._push_coalesced_scan_random_list
//                (TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level);
//			global_barrier.sync_grid_opt();
//
//			if(mdata.worklist_sz_sml[0] + 
//					mdata.worklist_sz_mid[0] + 
//					mdata.worklist_sz_lrg[0] == 0) break;
//			
//			for(index_t i = TID; i < mdata.worklist_sz_sml[0]; i += GRNTY)
//			{
//				vertex_t frontier = mdata.worklist_sml[i];
//				index_t beg = ggraph.beg_pos[frontier];
//				index_t end = ggraph.beg_pos[frontier+1];
//
//				for(index_t j = beg; j < end; j ++)
//				{
//					vertex_t vert_end = ggraph.adj_list[j];
//#ifdef __AGG_MIN__
//					weight = ggraph.weight_list[j];
//#endif
//					feature_t dist = (*(compute_mapper.edge_compute_push))(frontier,vert_end,
//							level,ggraph.beg_pos,weight,mdata.vert_status, mdata.vert_status_prev);
//#ifdef __VOTE__
//					if(mdata.vert_status[vert_end] != dist)
//					{
//						mdata.vert_status[vert_end] = dist;
//						appr_work += ggraph.beg_pos[vert_end + 1] - ggraph.beg_pos[vert_end];
//					}
//#elif __AGG_MIN__
//					if(mdata.vert_status[vert_end] > dist)
//					{
//						atomicMin(mdata.vert_status + vert_end, dist);
//						appr_work += ggraph.beg_pos[vert_end + 1] - ggraph.beg_pos[vert_end];
//					}
//#elif __AGG_SUB__
//					if(mdata.vert_status[vert_end] > K)
//					{
//						atomicSub(mdata.vert_status + vert_end, dist);
//						//appr_work += ggraph.beg_pos[vert_end + 1] - ggraph.beg_pos[vert_end];
//					}
//#endif
//				}
//			}
//			
//			
//			for(index_t i = wid_in_grd; i < mdata.worklist_sz_mid[0]; i += WGRNTY)
//			{
//				vertex_t frontier = mdata.worklist_mid[i];
//				index_t beg = ggraph.beg_pos[frontier];
//				index_t end = ggraph.beg_pos[frontier+1];
//
//				for(index_t j = beg + tid_in_wrp; j < end; j += 32)
//				{
//					vertex_t vert_end = ggraph.adj_list[j];
//#ifdef __AGG_MIN__
//					weight = ggraph.weight_list[j];
//#endif
//					feature_t dist = (*(compute_mapper.edge_compute_push))(frontier,vert_end,
//							level,ggraph.beg_pos,weight,mdata.vert_status, mdata.vert_status_prev);
//#ifdef __VOTE__
//					if(mdata.vert_status[vert_end] != dist)
//					{
//						mdata.vert_status[vert_end] = dist;
//						appr_work += ggraph.beg_pos[vert_end + 1] - ggraph.beg_pos[vert_end];
//					}
//#elif __AGG_MIN__
//					if(mdata.vert_status[vert_end] > dist)
//					{
//						atomicMin(mdata.vert_status + vert_end, dist);
//						appr_work += ggraph.beg_pos[vert_end + 1] - ggraph.beg_pos[vert_end];
//					}
//#elif __AGG_SUB__
//					if(mdata.vert_status[vert_end] > K)
//					{
//						atomicSub(mdata.vert_status + vert_end, dist);
//						//appr_work += ggraph.beg_pos[vert_end + 1] - ggraph.beg_pos[vert_end];
//					}
//#endif
//				}
//			}
//			
//			
//			for(index_t i = blockIdx.x; i < mdata.worklist_sz_lrg[0]; i += gridDim.x)
//			{
//				vertex_t frontier = mdata.worklist_lrg[i];
//				index_t beg = ggraph.beg_pos[frontier];
//				index_t end = ggraph.beg_pos[frontier+1];
//
//				for(index_t j = beg + threadIdx.x; j < end; j += blockDim.x)
//				{
//					vertex_t vert_end = ggraph.adj_list[j];
//#ifdef __AGG_MIN__
//					weight = ggraph.weight_list[j];
//#endif
//					feature_t dist = (*(compute_mapper.edge_compute_push))(frontier,vert_end,
//							level,ggraph.beg_pos,weight,mdata.vert_status, mdata.vert_status_prev);
//#ifdef __VOTE__
//					if(mdata.vert_status[vert_end] != dist)
//					{
//						mdata.vert_status[vert_end] = dist;
//						appr_work += ggraph.beg_pos[vert_end + 1] - ggraph.beg_pos[vert_end];
//					}
//#elif __AGG_MIN__
//					if(mdata.vert_status[vert_end] > dist)
//					{
//						atomicMin(mdata.vert_status + vert_end, dist);
//						appr_work += ggraph.beg_pos[vert_end + 1] - ggraph.beg_pos[vert_end];
//					}
//#elif __AGG_SUB__
//					if(mdata.vert_status[vert_end] > K)
//					{
//						atomicSub(mdata.vert_status + vert_end, dist);
//						//appr_work += ggraph.beg_pos[vert_end + 1] - ggraph.beg_pos[vert_end];
//					}
//#endif
//				}
//			}
//			if(level==1) is_topdown = false;
//		}
//		else
//		{
//			worklist_gather._pull_coalesced_scan_sorted_list
//                (TID, wid_in_blk, tid_in_wrp, wcount_in_blk, GRNTY, level);
//			global_barrier.sync_grid_opt();
//			
//			if( (mdata.worklist_sz_sml[0] + 
//					mdata.worklist_sz_mid[0] + 
//					mdata.worklist_sz_lrg[0] == 0) ||
//					(fq_sz == (mdata.worklist_sz_sml[0] + 
//					mdata.worklist_sz_mid[0] + 
//					mdata.worklist_sz_lrg[0]))) break;
//					
//			fq_sz = mdata.worklist_sz_sml[0] + 
//					mdata.worklist_sz_mid[0] + 
//					mdata.worklist_sz_lrg[0];
//			for(index_t i = TID; i < mdata.worklist_sz_sml[0]; i += GRNTY)
//			{
//				vertex_t frontier = mdata.worklist_sml[i];
//				index_t beg = ggraph.beg_pos[frontier];
//				index_t end = ggraph.beg_pos[frontier+1];
//
//#ifdef  __AGG_SUM__
//					feature_t frontier_vert_status = 0;
//#elif 	__AGG_SUB__
//					feature_t frontier_vert_status = 0;
//#elif 	__AGG_MIN__
//					feature_t frontier_vert_status = INFTY;
//#endif
//					for(index_t j = beg; j < end; j ++)
//					{
//						vertex_t vert_src = ggraph.adj_list[j];
//#ifdef __AGG_MIN__
//						weight_t weight = ggraph.weight_list[j];
//#endif
//						feature_t dist = (*(compute_mapper.edge_compute_pull))(vert_src, frontier,
//								level,ggraph.beg_pos,weight,mdata.vert_status, 
//								mdata.vert_status_prev);
//#ifdef __VOTE__
//						if(dist == level)
//						{
//							mdata.vert_status[frontier] = level + 1;
//							break;
//						}
//#elif __AGG_SUM__
//						frontier_vert_status += dist;
//#elif __AGG_SUB__
//						frontier_vert_status -= dist;
//#elif __AGG_MIN__
//						if(frontier_vert_status > dist) frontier_vert_status = dist;
//#endif
//					}
//#ifdef __AGG_SUM__
//					mdata.vert_status[frontier] = (0.15 + 0.85 * frontier_vert_status)
//						/(ggraph.beg_pos[frontier + 1] - ggraph.beg_pos[frontier]);
//#elif  __AGG_SUB__
//					mdata.vert_status[frontier] -= frontier_vert_status;
//#elif __AGG_MIN__
//					if(mdata.vert_status[frontier] > frontier_vert_status) 
//						mdata.vert_status[frontier] = frontier_vert_status;
//#endif
//			}
//
//			for(index_t i = wid_in_grd; i < mdata.worklist_sz_mid[0]; i += WGRNTY)
//			{
//				vertex_t frontier = mdata.worklist_mid[i];
//				index_t beg = ggraph.beg_pos[frontier];
//					index_t end = ggraph.beg_pos[frontier+1];
//#ifdef __AGG_SUM__
//					feature_t frontier_vert_status=0;
//#elif __AGG_SUB__
//					feature_t frontier_vert_status=0;
//#elif __AGG_MIN__
//					feature_t frontier_vert_status=INFTY;
//#endif
//					for(index_t j = beg + tid_in_wrp; __any(j < end); j += 32)
//					{
//						feature_t dist = INFTY;
//						if(j<end)
//						{
//							vertex_t vert_src=ggraph.adj_list[j];
//#ifdef __AGG_MIN__
//							weight_t weight = ggraph.weight_list[j];
//#endif
//							dist = (*(compute_mapper.edge_compute_pull))(vert_src, frontier,
//								level,ggraph.beg_pos,weight,mdata.vert_status, 
//								mdata.vert_status_prev);
//						}
//#ifdef __VOTE__ 		
//						int predicate = (dist == level) * (j < end);
//						if(__any(predicate))
//						{
//							if(!tid_in_wrp) mdata.vert_status[frontier] = level + 1;
//							break;
//						}
//#elif __AGG_SUM__
//						frontier_vert_status += dist;
//#elif __AGG_SUB__
//						frontier_vert_status -= dist;
//#elif __AGG_MIN__
//						if(frontier_vert_status > dist) frontier_vert_status = dist;
//#endif
//					}
//#ifdef __AGG_SUM__
//					for (int j=16; j>=1; j>>=1)
//						frontier_vert_status += __shfl_xor(frontier_vert_status, j, 32);
//					if(!tid_in_wrp)
//						mdata.vert_status[frontier] = (0.15 + 0.85*frontier_vert_status)
//							/(ggraph.beg_pos[frontier+1]-ggraph.beg_pos[frontier]);
//#elif __AGG_MIN__
//					feature_t tmp;
//					for (int j=16; j>=1; j>>=1)
//					{
//						tmp = __shfl_xor(frontier_vert_status, j, 32);
//						if(frontier_vert_status > tmp) frontier_vert_status = tmp;
//					}
//					
//					if(!tid_in_wrp) 
//						if( mdata.vert_status[frontier] > frontier_vert_status)
//							mdata.vert_status[frontier] = frontier_vert_status;
//
//#elif __AGG_SUB__
//					for (int j=16; j>=1; j>>=1)
//						frontier_vert_status += __shfl_xor(frontier_vert_status, j, 32);
//					if(!tid_in_wrp)
//						mdata.vert_status[frontier]+=frontier_vert_status;
//#endif
//				}
//				
//			for(index_t i = blockIdx.x; i < mdata.worklist_sz_lrg[0]; i += gridDim.x)
//			{
//				vertex_t frontier = mdata.worklist_lrg[i];
//					index_t beg=ggraph.beg_pos[frontier];
//					index_t end=ggraph.beg_pos[frontier+1];
//#ifdef  __AGG_SUM__
//					feature_t frontier_vert_status=0;
//#elif  __AGG_SUB__
//					feature_t frontier_vert_status=0;
//#elif __AGG_MIN__
//					feature_t frontier_vert_status=INFTY;
//#endif
//					for(index_t j = beg + threadIdx.x; 
//							__syncthreads_or(j < end); j += blockDim.x)
//					{
//						feature_t dist = INFTY;
//						if(j < end)
//						{
//							vertex_t vert_src=ggraph.adj_list[j];
//#ifdef __AGG_MIN__
//							weight_t weight = ggraph.weight_list[j];
//#endif
//							dist =(*(compute_mapper.edge_compute_pull))(vert_src, frontier,
//								level,ggraph.beg_pos,weight,mdata.vert_status, 
//								mdata.vert_status_prev);
//						}
//						
//#ifdef __VOTE__ 
//						int predicate = (dist == level) * (j < end);
//						if(__syncthreads_or(predicate))
//						{
//							if(!threadIdx.x) mdata.vert_status[frontier]= level + 1;
//							break;
//						}
//#elif __AGG_SUM__
//						frontier_vert_status+=dist;
//#elif __AGG_SUB__
//						frontier_vert_status-=dist;
//#elif __AGG_MIN__
//						if(frontier_vert_status > dist) frontier_vert_status = dist;
//#endif
//					}
//
//#ifdef __AGG_SUM__
//					smem[threadIdx.x]=frontier_vert_status;
//					__syncthreads();
//					int idx=blockDim.x>>1;
//					while(idx)
//					{
//						if(threadIdx.x<idx)
//							smem[threadIdx.x]+=smem[threadIdx.x+idx];
//
//						__syncthreads();
//						idx>>=1;
//					}
//					__syncthreads();
//
//					if(threadIdx.x==0)
//						mdata.vert_status[frontier]=(0.15 + 0.85*smem[0])
//							/(ggraph.beg_pos[frontier+1]-ggraph.beg_pos[frontier]);
//#elif __AGG_MIN__
//					smem[threadIdx.x]=frontier_vert_status;
//					__syncthreads();
//					int idx=blockDim.x>>1;
//					while(idx)
//					{
//						if(threadIdx.x<idx)
//							if(smem[threadIdx.x] > smem[threadIdx.x+idx])
//								smem[threadIdx.x] = smem[threadIdx.x+idx];
//
//						__syncthreads();
//						idx>>=1;
//					}
//					__syncthreads();
//
//					if(threadIdx.x==0) 
//						if(mdata.vert_status[frontier] > smem[0])
//							mdata.vert_status[frontier] = smem[0];
//#elif __AGG_SUB__
//					smem[threadIdx.x]=frontier_vert_status;
//					__syncthreads();
//					int idx=blockDim.x>>1;
//					while(idx)
//					{
//						if(threadIdx.x<idx)
//							smem[threadIdx.x]+=smem[threadIdx.x+idx];
//
//						__syncthreads();
//						idx>>=1;
//					}
//					__syncthreads();
//
//					if(threadIdx.x==0)
//						mdata.vert_status[frontier]+=smem[0];
//#endif
//				}
//			//if(level==4) break;
//			//break;
//			if(level==4) is_topdown = true;
//		}
//		
//		global_barrier.sync_grid_opt();
//		
//		if(TID == 0)  mdata.worklist_sz_sml[0] = 0;
//		if(TID == 32) mdata.worklist_sz_mid[0] = 0;
//		if(TID == 64) mdata.worklist_sz_lrg[0] = 0;
//		
//		level ++;
//	}
//	if(!TID) level_record[0] = level;
//}
//


__global__ void
mapper_bin_push_kernel(
	feature_t level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper
){
	
    const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;
	//const index_t wid_in_blk = threadIdx.x >> 5;
	const index_t wid_in_grd = TID >> 5;
	//const index_t wcount_in_blk = blockDim.x >> 5;
	const index_t WGRNTY = GRNTY >> 5;
	const index_t BIN_OFF = TID * BIN_SZ;
    if(TID == 0) mdata.worklist_sz_sml[0] = 0;

    vertex_t my_bin_sz = 0;
    index_t appr_work = 0;
    compute_mapper.mapper_bin_push(
            appr_work,
            mdata.worklist_sz_sml,
            my_bin_sz,
            mdata.worklist_bin,
            mdata.worklist_sz_mid[0],
            mdata.worklist_mid,
            wid_in_grd,
            32,
            WGRNTY,
            tid_in_wrp,
            level,
            BIN_OFF);
    
    assert(mdata.worklist_sz_sml[0] != -1);

    mdata.cat_thd_count_mid[TID] = my_bin_sz;
}

__global__ void
mapper_push_sml(
	feature_t level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper
){
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;

	compute_mapper.mapper_push(
			mdata.worklist_sz_sml[0],
			mdata.worklist_sml,
			mdata.cat_thd_count_sml,
			TID,/*group id*/
			1,/*group size*/
			GRNTY,/*group count*/
			0,/*thread off intra group*/
			level);
}

__global__ void
mapper_push_mid(
	feature_t level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper
){
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;
	const index_t wid_in_grd = TID >> 5;
	const index_t WGRNTY = GRNTY >> 5;

	compute_mapper.mapper_push(
			mdata.worklist_sz_mid[0],
			mdata.worklist_mid,
			mdata.cat_thd_count_mid,
			wid_in_grd,/*group id*/
			32,/*group size*/
			WGRNTY,/*group count*/
			tid_in_wrp,/*thread off intra group*/
			level);
}

__global__ void
mapper_push_lrg(
	feature_t level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper
){
	compute_mapper.mapper_push(
			mdata.worklist_sz_lrg[0],
			mdata.worklist_lrg,
			mdata.cat_thd_count_lrg,
			blockIdx.x,/*group id*/
			blockDim.x,/*group size*/
			gridDim.x,/*group count*/
			threadIdx.x,/*thread off intra group*/
			level);
}

__global__ void
mapper_pull_sml(
	feature_t level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper
){
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	
	compute_mapper.thd_mapper_pull(
			mdata.cat_thd_count_sml,
			mdata.worklist_sz_sml[0], 
			mdata.worklist_sml, 
			TID, 
			GRNTY, 
			level);
}

__global__ void
mapper_pull_mid(
	feature_t level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper
){
	const index_t TID = threadIdx.x + blockIdx.x * blockDim.x;
	const index_t GRNTY = blockDim.x * gridDim.x;
	const index_t tid_in_wrp = threadIdx.x & 31;
	const index_t wid_in_grd = TID >> 5;
	const index_t WGRNTY = GRNTY >> 5;
	
	compute_mapper.warp_mapper_pull(
			mdata.cat_thd_count_mid,
			mdata.worklist_sz_mid[0], 
			mdata.worklist_mid, 
			wid_in_grd, 
			tid_in_wrp, 
			WGRNTY, 
			level);
}

/* Called by user 
 * Push model
 */
__global__ void
mapper_pull_lrg(
		feature_t level,
		gpu_graph ggraph,
		meta_data mdata,
		mapper compute_mapper
){
	compute_mapper.cta_mapper_pull(
			mdata.cat_thd_count_lrg,
			mdata.worklist_sz_lrg[0], 
			mdata.worklist_lrg, 
			level);
}

int balanced_push(
    int cfg_blk_size,
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier){
	
	int blk_size = 0;
	int grd_size = 0;
	//cudaFuncGetAttributes
	cudaOccupancyMaxPotentialBlockSize(&grd_size, &blk_size, 
			balanced_push_kernel, 0, 0);
    
	grd_size = (blk_size * grd_size)/ cfg_blk_size;
	blk_size = cfg_blk_size;
	//grd_size = (blk_size * grd_size)/ 128;
	//blk_size = 128;

        printf("balanced push-- block=%d, grid=%d\n", blk_size, grd_size);
	assert(blk_size*grd_size <= BLKS_NUM*THDS_NUM);

	//push_pull_opt_kernel
	balanced_push_kernel
		<<<grd_size, blk_size>>>
		(level, 
		 ggraph, 
		 mdata, 
		 compute_mapper, 
		 worklist_gather,
		 global_barrier);
	H_ERR(cudaThreadSynchronize());

    //cudaMemcpy(mdata.sml_count_chk, mdata.cat_thd_count_sml, sizeof(index_t)*blk_size*grd_size, cudaMemcpyDeviceToHost);
    //cudaMemcpy(mdata.mid_count_chk, mdata.cat_thd_count_mid, sizeof(index_t)*blk_size*grd_size, cudaMemcpyDeviceToHost);
    //cudaMemcpy(mdata.lrg_count_chk, mdata.cat_thd_count_lrg, sizeof(index_t)*blk_size*grd_size, cudaMemcpyDeviceToHost);
	//
    //index_t total_count = 0;
    //for(int i = 0; i < blk_size * grd_size; i++)
    //    total_count+=mdata.sml_count_chk[i] + mdata.mid_count_chk[i] + mdata.lrg_count_chk[i];
    //
    //printf("---debug total count: %ld\n", total_count);
	return 0;
}


int push_pull_opt(
        int iter_limit,
        bool is_topdown_input,
        int cfg_blk_size,
        feature_t *level,
        gpu_graph ggraph,
        meta_data mdata,
        mapper compute_mapper,
        reducer worklist_gather,
        Barrier global_barrier){
	
	int blk_size = 0;
	int grd_size = 0;
	//cudaFuncGetAttributes
	cudaOccupancyMaxPotentialBlockSize(&grd_size, &blk_size, 
			push_pull_fused_kernel, 0, 0);

	grd_size = (blk_size * grd_size)/ cfg_blk_size;
	blk_size = cfg_blk_size;
	//grd_size = (blk_size * grd_size)/ 128;
	//blk_size = 128;

	printf("optimal -- block=%d, grid=%d\n", blk_size, grd_size);
	assert(blk_size*grd_size <= BLKS_NUM*THDS_NUM);

	push_pull_fused_kernel
		<<<grd_size, blk_size>>>
		(iter_limit,
         is_topdown_input,
         level, 
		 ggraph, 
		 mdata, 
		 compute_mapper, 
		 worklist_gather,
		 global_barrier);
	H_ERR(cudaThreadSynchronize());
	return 0;
}

int mapper_hybrid_push_merge(
    int cfg_blk_size,
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier){
	
	int blk_size = 0;
	int grd_size = 0;
	cudaOccupancyMaxPotentialBlockSize(&grd_size, &blk_size, 
			hybrid_bin_scan_push_kernel, 0, 0);

	grd_size = (blk_size * grd_size)/ cfg_blk_size;
	blk_size = cfg_blk_size;

	printf("merge -- block=%d, grid=%d\n", blk_size, grd_size);
	assert(blk_size*grd_size <= BLKS_NUM*THDS_NUM);

    hybrid_bin_scan_push_kernel	
		<<<grd_size, blk_size>>>
		(level, 
		 ggraph, 
		 mdata, 
		 compute_mapper, 
		 worklist_gather,
		 global_barrier);
	H_ERR(cudaThreadSynchronize());
	return 0;
}

int mapper_merge_push(
    int cfg_blk_size,
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier){
	
	int blk_size = 0;
	int grd_size = 0;
	cudaOccupancyMaxPotentialBlockSize(&grd_size, &blk_size, 
			merge_push_kernel, 0, 0);

	grd_size = (blk_size * grd_size)/ cfg_blk_size;
	blk_size = cfg_blk_size;

	printf("merge -- block=%d, grid=%d\n", blk_size, grd_size);
	assert(blk_size*grd_size <= BLKS_NUM*THDS_NUM);

	merge_push_kernel
		<<<grd_size, blk_size>>>
		(level, 
		 ggraph, 
		 mdata, 
		 compute_mapper, 
		 worklist_gather,
		 global_barrier);
	H_ERR(cudaThreadSynchronize());


    //cudaMemcpy(mdata.sml_count_chk, mdata.cat_thd_count_sml, sizeof(index_t)*blk_size*grd_size, cudaMemcpyDeviceToHost);
//    cudaMemcpy(mdata.mid_count_chk, mdata.cat_thd_count_mid, sizeof(index_t)*blk_size*grd_size, cudaMemcpyDeviceToHost);
//    //cudaMemcpy(mdata.lrg_count_chk, mdata.cat_thd_count_lrg, sizeof(index_t)*blk_size*grd_size, cudaMemcpyDeviceToHost);
//	
//    index_t total_count = 0;
//    for(int i = 0; i < blk_size * grd_size; i++)
//        //total_count+=mdata.sml_count_chk[i] + mdata.mid_count_chk[i] + mdata.lrg_count_chk[i];
//        total_count+=mdata.mid_count_chk[i];
//    
//    printf("---debug total count: %ld\n", total_count);
	return 0;
}

int mapper_merge_pull(
    int cfg_blk_size,
	feature_t terminate_level,
	feature_t *level,
	gpu_graph ggraph,
	meta_data mdata,
	mapper compute_mapper,
	reducer worklist_gather,
	Barrier global_barrier){
	
	int blk_size = 0;
	int grd_size = 0;
	cudaOccupancyMaxPotentialBlockSize(&grd_size, &blk_size, 
			merge_pull_kernel, 0, 0);

	grd_size = (blk_size * grd_size)/ cfg_blk_size;
	blk_size = cfg_blk_size;

	printf("merge -- block=%d, grid=%d\n", blk_size, grd_size);
	assert(blk_size*grd_size <= BLKS_NUM*THDS_NUM);

	merge_pull_kernel
		<<<grd_size, blk_size>>>
		(terminate_level,
		 level, 
		 ggraph, 
		 mdata, 
		 compute_mapper, 
		 worklist_gather,
		 global_barrier);
	H_ERR(cudaThreadSynchronize());
	return 0;
}

/* Called by user
 * Push model
 */
int mapper_push(
		feature_t level,
		gpu_graph ggraph,
		meta_data mdata,
		mapper compute_mapper
 ){
	mapper_push_sml<<<BLKS_NUM,THDS_NUM,0, mdata.stream[0]>>>
		(level,ggraph,mdata,compute_mapper);

	mapper_push_mid<<<BLKS_NUM,THDS_NUM,0, mdata.stream[1]>>>
		(level,ggraph,mdata,compute_mapper);

	mapper_push_lrg<<<BLKS_NUM,THDS_NUM,0, mdata.stream[2]>>>
		(level,ggraph,mdata,compute_mapper);

	H_ERR(cudaStreamSynchronize(mdata.stream[0]));
	H_ERR(cudaStreamSynchronize(mdata.stream[1]));
	H_ERR(cudaStreamSynchronize(mdata.stream[2]));

	return 0;
}


/* Called by user
 * Pull model
 */
int mapper_pull(
		feature_t level,
		gpu_graph ggraph,
		meta_data mdata,
		mapper compute_mapper
 ){
	mapper_pull_sml<<<BLKS_NUM,THDS_NUM,0, mdata.stream[0]>>>
		(level,ggraph,mdata,compute_mapper);

	mapper_pull_mid<<<BLKS_NUM,THDS_NUM,0, mdata.stream[1]>>>
		(level,ggraph,mdata,compute_mapper);

	mapper_pull_lrg<<<BLKS_NUM,THDS_NUM,0, mdata.stream[2]>>>
		(level,ggraph,mdata,compute_mapper);

	H_ERR(cudaStreamSynchronize(mdata.stream[0]));
	H_ERR(cudaStreamSynchronize(mdata.stream[1]));
	H_ERR(cudaStreamSynchronize(mdata.stream[2]));

	return 0;
}

void mapper_push_bin_gather(
        feature_t *level,
        gpu_graph ggraph,
        meta_data mdata,
        mapper compute_mapper,
        reducer reducer_inst){
    int blk_count = 208;
    int thd_count = 128;
    feature_t level_h = 0;

    vertex_t *mid;
    cudaMallocHost((void **)&mid, sizeof(vertex_t));
    H_ERR(cudaMemcpy(mid, (vertex_t *)mdata.worklist_sz_mid, 
           sizeof(vertex_t), cudaMemcpyDeviceToHost));
    
    //std::cout<<"Traversed "<< level_h<< " " <<mid[0]<<"\n";
    
    while(true){
       mapper_bin_push_kernel<<<blk_count, thd_count>>>(
        level_h,ggraph, mdata, compute_mapper);
       H_ERR(cudaThreadSynchronize()); 
    
       H_ERR(cudaMemset((vertex_t *)mdata.worklist_sz_mid, 0, sizeof(vertex_t)));
       H_ERR(cudaThreadSynchronize());

       grid_scan<vertex_t, index_t><<<blk_count, thd_count>>>(
        mdata);
       H_ERR(cudaThreadSynchronize()); 

       H_ERR(cudaMemcpy(mid, (vertex_t *)mdata.worklist_sz_mid, 
           sizeof(vertex_t), cudaMemcpyDeviceToHost));
       
       if(mid[0] == 0) break;
       //std::cout<<"Traversed "<< level_h<< " " <<mid[0]<<"\n";

       thread_stride_gather<<<blk_count, thd_count>>>(
        level_h,ggraph, mdata, reducer_inst);
       H_ERR(cudaThreadSynchronize()); 

       (level_h) ++;
    }
    std::cout<<"Traversed "<< level_h<< " levels\n";
}

#endif
