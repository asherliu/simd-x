#ifndef __MAPPER__
#define __MAPPER__
#include "header.h"
#include "util.h"
#include "gpu_graph.cuh"
#include "meta_data.cuh"
#include <assert.h>
typedef feature_t (*cb_mapper)
(	vertex_t 		active_edge_src,
	vertex_t		active_edge_end,
	feature_t		level,
	index_t*		beg_pos,
	weight_t		weight_list,
	feature_t* 		vert_status,
	feature_t* 		vert_status_prev);


//Bitmap could also be helpful
/* mapper kernel function */
class mapper
{
	public:
		//variable
		vertex_t *adj_list;
		weight_t* weight_list;
		index_t *beg_pos;
		index_t vert_count;
		feature_t *vert_status;
		feature_t *vert_status_prev;

		//index_t *cat_thd_count_sml;
		cb_mapper edge_compute; 
		cb_mapper edge_compute_push; 
		cb_mapper edge_compute_pull; 

	public:
		//constructor
		mapper(gpu_graph ggraph, meta_data mdata,
				cb_mapper user_mapper_push,
				cb_mapper user_mapper_pull)
		{
			adj_list = ggraph.adj_list;
			weight_list = ggraph.weight_list;
			beg_pos = ggraph.beg_pos;
			vert_count = ggraph.vert_count;
			vert_status = mdata.vert_status;
			vert_status_prev = mdata.vert_status_prev;
			//cat_thd_count_sml = mdata.cat_thd_count_sml;

			edge_compute_push = user_mapper_push;
			edge_compute_pull = user_mapper_pull;
		}

		~mapper(){}

	public:
		//function
		//Could represent thread, warp, cta, grid grained thread scheduling
		
		__forceinline__  __device__ void
			mapper_push(
					vertex_t wqueue,
					vertex_t *worklist,
					index_t *cat_thd_count,
					const index_t GRP_ID,
					const index_t GRP_SZ,
					const index_t GRP_COUNT,
					const index_t THD_OFF,
					feature_t level)
			{
				index_t appr_work = 0;
				weight_t weight;
				const vertex_t WSZ = wqueue;
				for(index_t i = GRP_ID; i < WSZ; i += GRP_COUNT)
				{
					vertex_t frontier = worklist[i];
					index_t beg = beg_pos[frontier];
					index_t end = beg_pos[frontier+1];

					for(index_t j = beg + THD_OFF; j < end; j += GRP_SZ)
					{
						vertex_t vert_end = adj_list[j];
						weight = weight_list[j];
						feature_t dist = (*edge_compute_push)(frontier,vert_end,
								level,beg_pos,weight,vert_status, vert_status_prev);
#ifdef __VOTE__
						if(vert_status[vert_end] != dist)
						{
							vert_status[vert_end] = dist;
							appr_work += beg_pos[vert_end + 1] - beg_pos[vert_end];
						}
#elif __AGG_MIN__
						if(vert_status[vert_end] > dist)
						{
							if(atomicMin(vert_status + vert_end, dist) > dist)
                            //vert_status[vert_end] = dist;
    							appr_work += beg_pos[vert_end + 1] - beg_pos[vert_end];
						}
#elif __AGG_MIN_ATOMIC__ //Collect compute-combine benefits over compute-direct-update
						if(vert_status[vert_end] > dist)
						{
							if(atomicMin(vert_status + vert_end, dist) > dist)
    							appr_work += beg_pos[vert_end + 1] - beg_pos[vert_end];
						}

#elif __AGG_SUB__
						if(vert_status[vert_end] > K)
						{
							if (atomicSub(vert_status + vert_end, dist) == K)
    							appr_work += beg_pos[vert_end + 1] - beg_pos[vert_end];
						}
#endif
					}
				}

				//note, we use cat_thd_count to store the future amount of workload
				//and such data is important for switching between push - pull models.
				cat_thd_count[threadIdx.x + blockIdx.x * blockDim.x] = appr_work;
			}

		__device__ __forceinline__  void
			mapper_bin_push_online_alone(
					index_t &appr_work,
                    vertex_t &my_front_count,
					vertex_t *worklist_bin,
					vertex_t wqueue,
					vertex_t *worklist,
					const index_t GRP_ID,
					const index_t GRP_SZ,
					const index_t GRP_COUNT,
					const index_t THD_OFF,
					feature_t level,
					index_t bin_off)
			{
				const vertex_t WSZ = wqueue;
				weight_t weight;

				for(index_t i = GRP_ID;i < WSZ; i += GRP_COUNT)
				{
					vertex_t frontier = worklist[i];
					index_t beg = beg_pos[frontier];
					index_t end = beg_pos[frontier+1];

					for(index_t j = beg + THD_OFF; j < end; j += GRP_SZ)
					{
						vertex_t vert_end = adj_list[j];
#ifdef __AGG_MIN__
						weight_t weight = weight_list[j];
#endif
						feature_t dist = (*edge_compute_push)(frontier,vert_end,
								level,beg_pos,weight,vert_status, vert_status_prev);
#ifdef __VOTE__	
						if(vert_status[vert_end] != dist)
						{
							vert_status[vert_end] = dist;
				            if(my_front_count < BIN_SZ)
                            {
                                worklist_bin[bin_off + my_front_count] = vert_end;
                                my_front_count ++;
	//						    appr_work += beg_pos[vert_end + 1] - beg_pos[vert_end];
                            }
                            assert(my_front_count < BIN_SZ);
						}
#elif __AGG_MIN__
						if(vert_status[vert_end] > dist)
						{
                            //atomicMin does not return mininum
                            //instead the old val, in this case vert_status[vert_end].
                             
                            if(atomicMin(vert_status + vert_end, dist)> dist)
                                if(my_front_count < BIN_SZ)
                                {
                                    worklist_bin[bin_off + my_front_count] = vert_end;
                                    my_front_count ++;
	//						        appr_work += beg_pos[vert_end + 1] - beg_pos[vert_end];
                                }
                            assert(my_front_count < BIN_SZ);
						}
#elif __AGG_MIN_ATOMIC__ //Collect compute-combine benefits over compute-direct-update
						if(vert_status[vert_end] > dist)
						{
                            //atomicMin does not return mininum
                            //instead the old val, in this case vert_status[vert_end].
                            if(atomicMin(vert_status + vert_end, dist)> dist)
                                if(my_front_count < BIN_SZ)
                                {
                                    worklist_bin[bin_off + my_front_count] = vert_end;
                                    my_front_count ++;
	//						        appr_work += beg_pos[vert_end + 1] - beg_pos[vert_end];
                                }
                            assert(my_front_count < BIN_SZ);
						}
#elif __AGG_SUB__
						if(vert_status[vert_end] > K)
						{
							if(atomicSub(vert_status + vert_end, dist) == K + 1)
				                if(my_front_count < BIN_SZ)
                                {
                                    worklist_bin[bin_off+my_front_count]=vert_end;
							        my_front_count ++;
	//						        appr_work += beg_pos[vert_end + 1] - beg_pos[vert_end];
                                }else assert(my_front_count < BIN_SZ);
                        }
#endif
					}
				}
			}

		__device__ __forceinline__  void
			mapper_bin_push(
					index_t &appr_work,
                    volatile vertex_t *overflow_indicator,
                    vertex_t &my_front_count,
					vertex_t *worklist_bin,
					vertex_t wqueue,
					vertex_t *worklist,
					const index_t GRP_ID,
					const index_t GRP_SZ,
					const index_t GRP_COUNT,
					const index_t THD_OFF,
					feature_t level,
					index_t bin_off)
			{
				const vertex_t WSZ = wqueue;
				weight_t weight;

				for(index_t i = GRP_ID;i < WSZ; i += GRP_COUNT)
				{
					vertex_t frontier = worklist[i];
					index_t beg = beg_pos[frontier];
					index_t end = beg_pos[frontier+1];

					for(index_t j = beg + THD_OFF; j < end; j += GRP_SZ)
					{
						vertex_t vert_end = adj_list[j];
						weight_t weight = weight_list[j];
						feature_t dist = (*edge_compute_push)(frontier,vert_end,
								level,beg_pos,weight,vert_status, vert_status_prev);
#ifdef __VOTE__	
						if(vert_status[vert_end] != dist)
						{
							vert_status[vert_end] = dist;
				            if(my_front_count < BIN_SZ)
                            {
                                worklist_bin[bin_off + my_front_count] = vert_end;
                                my_front_count ++;
							    appr_work += beg_pos[vert_end + 1] - beg_pos[vert_end];
                            }
                            //assert(my_front_count < BIN_SZ);
						}
#elif __AGG_MIN__
						if(vert_status[vert_end] > dist)
						{
                            //atomicMin does not return mininum
                            //instead the old val, in this case vert_status[vert_end].
                            if(atomicMin(vert_status + vert_end, dist)> dist)
                                if(my_front_count < BIN_SZ)
                                {
                                    worklist_bin[bin_off + my_front_count] = vert_end;
                                    my_front_count ++;
							        appr_work += beg_pos[vert_end + 1] - beg_pos[vert_end];
                                }
                                else overflow_indicator[0] = -1;
						}
#elif __AGG_MIN_ATOMIC__ //Collect compute-combine benefits over compute-direct-update
						if(vert_status[vert_end] > dist)
						{
                            //atomicMin does not return mininum
                            //instead the old val, in this case vert_status[vert_end].
                            if(atomicMin(vert_status + vert_end, dist)> dist)
                                if(my_front_count < BIN_SZ)
                                {
                                    worklist_bin[bin_off + my_front_count] = vert_end;
                                    my_front_count ++;
							        appr_work += beg_pos[vert_end + 1] - beg_pos[vert_end];
                                }
                                else overflow_indicator[0] = -1;
						}
#elif __AGG_SUB__
						if(vert_status[vert_end] > K)
						{
							if(atomicSub(vert_status + vert_end, dist) == K + 1)
				                if(my_front_count < BIN_SZ)
                                {
                                    worklist_bin[bin_off+my_front_count]=vert_end;
							        my_front_count ++;
							        appr_work += beg_pos[vert_end + 1] - beg_pos[vert_end];
                                }
                                else overflow_indicator[0] = -1;
                        }
#endif
					}
				}

				//Make sure NO overflow!
				if(my_front_count >= BIN_SZ)
                    overflow_indicator[0] = -1;
			}

		//Pull model function call
		/* thread pull model */
		__forceinline__  __device__ void
			thd_mapper_pull(
                    index_t *cat_thd_count,
					vertex_t wqueue,
					vertex_t *worklist,
					const index_t TID,
					const index_t GRNTY,
					feature_t level)
			{
				const vertex_t WSZ = wqueue;
				index_t appr_work = 0;
				for(index_t i = TID; i < WSZ; i += GRNTY)
				{
					vertex_t frontier = worklist[i];
					index_t beg = beg_pos[frontier];
					index_t end = beg_pos[frontier+1];
					feature_t frontier_vert_status = INFTY;
					for(index_t j = beg; j < end; j ++)
					{
						vertex_t vert_src = adj_list[j];
						weight_t weight = weight_list[j];
						feature_t dist = (*edge_compute_pull)(vert_src, frontier,
								level,beg_pos,weight,vert_status, vert_status_prev);
#ifdef __VOTE__
						if(dist == level && vert_status[frontier] == INFTY)
						{
                            appr_work ++;
							vert_status[frontier] = level + 1;
							break;
						}
#elif __AGG_SUM_AND_RATIO_
						frontier_vert_status += dist;
#elif __AGG_SUM__
						frontier_vert_status += dist;
#elif __AGG_SUB__
						frontier_vert_status -= dist;
#elif __AGG_MIN__
						if(frontier_vert_status > dist) frontier_vert_status = dist;

#elif __AGG_MIN_ATOMIC__ //Collect compute-combine benefits over compute-direct-update
                        if(atomicMin(vert_status+frontier, dist) > dist) 
                        {
                            appr_work ++;
                        }
#endif
					}
#ifdef __AGG_SUM__
					vert_status[frontier] = frontier_vert_status;
                    appr_work ++;
#elif __AGG_SUM_AND_RATIO__					
                    vert_status[frontier] = (0.15 + 0.85 * frontier_vert_status)
						/(beg_pos[frontier + 1] - beg_pos[frontier]);
                    appr_work++;
#elif  __AGG_SUB__
					vert_status[frontier] -= frontier_vert_status;
                    appr_work ++;
#elif __AGG_MIN__
					if(vert_status[frontier] > frontier_vert_status) 
                    {
                        appr_work ++;
						vert_status[frontier] = frontier_vert_status;
                    }
#endif
				}
				//note, we use cat_thd_count to store the future amount of workload
				//and such data is important for switching between push - pull models.
				cat_thd_count[threadIdx.x + blockIdx.x * blockDim.x] = appr_work;

			}

		/* warp mapper pull */
		__forceinline__  __device__ void
			warp_mapper_pull(
                    index_t *cat_thd_count,
					vertex_t wqueue,
					vertex_t *worklist,
					const index_t WID,
					const index_t WOFF,
					const index_t WGRNTY,
					feature_t level)
			{
				const vertex_t WSZ = wqueue;
				index_t appr_work = 0;
				for(index_t i = WID; i < WSZ; i += WGRNTY)
				{
					vertex_t frontier = worklist[i];
					index_t beg = beg_pos[frontier];
					index_t end = beg_pos[frontier+1];
					feature_t frontier_vert_status=INFTY;
					
                    //-----------This is coded to test Gunrock drawbacks----------
                    //- Because gunrock does not defer udpates, it cannot do 
                    //- collaborative early termination.
                    //for(index_t j = beg + WOFF; j < end; j += 32)
                    //----------------

					for(index_t j = beg + WOFF; __any(j < end); j += 32)
					{
						feature_t dist = INFTY;
						if(j<end)
						{
							vertex_t vert_src=adj_list[j];
							weight_t weight = weight_list[j];
							dist = (*edge_compute_pull)(vert_src, frontier,
								level,beg_pos,weight,vert_status, vert_status_prev);
						}
#ifdef __VOTE__ 		
						int predicate = (dist == level) * (j < end);
                        //-----------This is coded to test Gunrock drawbacks----------
                        //- Because gunrock does not defer udpates, it cannot do 
                        //- collaborative early termination.
                        //if(predicate!=0)
                        //{
                        //    vert_status[frontier] = level +1;
                        //    appr_work ++;
                        //    break;
                        //}
                        //-----------------------------------

						if(__any(predicate))
						{
							if(!WOFF) vert_status[frontier] = level + 1;
							appr_work ++;
                            break;
						}
#elif __AGG_SUM__
						frontier_vert_status += dist;
#elif __AGG_SUM_AND_RATIO__
						frontier_vert_status += dist;
#elif __AGG_SUB__
						frontier_vert_status -= dist;
#elif __AGG_MIN__
						if(frontier_vert_status > dist) frontier_vert_status = dist;
#elif __AGG_MIN_ATOMIC__ //Collect compute-combine benefits over compute-direct-update
                        if(atomicMin(vert_status+frontier, dist) > dist) 
                        {
                            appr_work ++;
                        }
#endif
					}
#ifdef __AGG_SUM__
					for (int j=16; j>=1; j>>=1)
						frontier_vert_status += __shfl_xor(frontier_vert_status, j, 32);
					if(!WOFF)
                    {
                        vert_status[frontier] = frontier_vert_status;
                        appr_work ++;
                    }
#elif __AGG_SUM_AND_RATIO__					
                    for (int j=16; j>=1; j>>=1)
						frontier_vert_status += __shfl_xor(frontier_vert_status, j, 32);
					if(!WOFF)
                    {
                        vert_status[frontier] = (0.15 + 0.85*frontier_vert_status)
							/(beg_pos[frontier+1]-beg_pos[frontier]);
                        appr_work ++;
                    }
#elif __AGG_MIN__
					feature_t tmp;
					for (int j=16; j>=1; j>>=1)
					{
						tmp = __shfl_xor(frontier_vert_status, j, 32);
						if(frontier_vert_status > tmp) frontier_vert_status = tmp;
					}
					
					if(!WOFF) 
						if( vert_status[frontier] > frontier_vert_status)
                        {
                            vert_status[frontier] = frontier_vert_status;
                            appr_work ++;
                        }

#elif __AGG_SUB__
					for (int j=16; j>=1; j>>=1)
						frontier_vert_status += __shfl_xor(frontier_vert_status, j, 32);
					if(!WOFF)
                    {
						vert_status[frontier]+=frontier_vert_status;
                        appr_work ++;
                    }

#endif
				}
				//note, we use cat_thd_count to store the future amount of workload
				//and such data is important for switching between push - pull models.
				cat_thd_count[threadIdx.x + blockIdx.x * blockDim.x] = appr_work;
			}

		/*cta mapper pull model kernel function*/
		__forceinline__  __device__ void
			cta_mapper_pull(
                    index_t *cat_thd_count,
					vertex_t wqueue,
					vertex_t *worklist,
					feature_t level)
			{
				const vertex_t WSZ = wqueue;
				index_t appr_work = 0;
                assert(blockDim.x <= THDS_NUM);
				__shared__ feature_t smem[THDS_NUM];
				for(index_t i = blockIdx.x; i < WSZ; i += gridDim.x)
				{
					vertex_t frontier=worklist[i];
					index_t beg=beg_pos[frontier];
					index_t end=beg_pos[frontier+1];
					feature_t frontier_vert_status=INFTY;
                    //-----------This is coded to test Gunrock drawbacks----------
                    //- Because gunrock does not defer udpates, it cannot do 
                    //- collaborative early termination.
                    //for(index_t j = beg + threadIdx.x; j < end; j += blockDim.x)
                    //--------
					for(index_t j = beg + threadIdx.x; 
							__syncthreads_or(j < end); j += blockDim.x)
					{
						feature_t dist = INFTY;
						if(j < end)
						{
							vertex_t vert_src=adj_list[j];
							weight_t weight = weight_list[j];
							dist =(*edge_compute_pull)(vert_src, frontier,
								level,beg_pos,weight,vert_status, vert_status_prev);
						}
						
#ifdef __VOTE__ 
						int predicate = (dist == level) * (j < end);
                        //-----------This is coded to test Gunrock drawbacks----------
                        //- Because gunrock does not defer udpates, it cannot do 
                        //- collaborative early termination.
                        //if(predicate!=0)
                        //{
                        //    vert_status[frontier] = level +1;
                        //    appr_work ++;
                        //    break;
                        //}
					    //-------------------

						if(__syncthreads_or(predicate))
						{
                            if(!threadIdx.x) vert_status[frontier]= level + 1;
                            appr_work ++;
							break;
						}
#elif __AGG_SUM__
						frontier_vert_status+=dist;
#elif __AGG_SUM_AND_RATIO__
						frontier_vert_status+=dist;
#elif __AGG_SUB__
						frontier_vert_status-=dist;
#elif __AGG_MIN__
						if(frontier_vert_status > dist) frontier_vert_status = dist;
#elif __AGG_MIN_ATOMIC__ //Collect compute-combine benefits over compute-direct-update
                        if(atomicMin(vert_status+frontier, dist) > dist) 
                        {
                            appr_work ++;
                        }
#endif
					}

#ifdef __AGG_SUM__
					smem[threadIdx.x]=frontier_vert_status;
					__syncthreads();
					int idx=blockDim.x>>1;
					while(idx)
					{
						if(threadIdx.x<idx)
							smem[threadIdx.x]+=smem[threadIdx.x+idx];

						__syncthreads();
						idx>>=1;
					}
					__syncthreads();

					if(threadIdx.x==0)
                    {
                        vert_status[frontier]=smem[0];
                        appr_work ++;
                    }

#elif __AGG_SUM_AND_RATIO__
					smem[threadIdx.x]=frontier_vert_status;
					__syncthreads();
					int idx=blockDim.x>>1;
					while(idx)
					{
						if(threadIdx.x<idx)
							smem[threadIdx.x]+=smem[threadIdx.x+idx];

						__syncthreads();
						idx>>=1;
					}
					__syncthreads();

					if(threadIdx.x==0)
                    {
                        vert_status[frontier]=(0.15 + 0.85*smem[0])
							/(beg_pos[frontier+1]-beg_pos[frontier]);
                        appr_work ++;
                    }

#elif __AGG_MIN__
					smem[threadIdx.x]=frontier_vert_status;
					__syncthreads();
					int idx=blockDim.x>>1;
					while(idx)
					{
						if(threadIdx.x<idx)
							if(smem[threadIdx.x] > smem[threadIdx.x+idx])
								smem[threadIdx.x] = smem[threadIdx.x+idx];

						__syncthreads();
						idx>>=1;
					}
					__syncthreads();

					if(threadIdx.x==0) 
						if(vert_status[frontier] > smem[0])
                        {
                            vert_status[frontier] = smem[0];
                            appr_work ++;
                        }
#elif __AGG_SUB__
					smem[threadIdx.x]=frontier_vert_status;
					__syncthreads();
					int idx=blockDim.x>>1;
					while(idx)
					{
						if(threadIdx.x<idx)
							smem[threadIdx.x]+=smem[threadIdx.x+idx];

						__syncthreads();
						idx>>=1;
					}
					__syncthreads();

					if(threadIdx.x==0)
                    {
                        vert_status[frontier]+=smem[0];
                        appr_work ++;
                    }
#endif
				}
				//note, we use cat_thd_count to store the future amount of workload
				//and such data is important for switching between push - pull models.
				cat_thd_count[threadIdx.x + blockIdx.x * blockDim.x] = appr_work;
			}
};

#endif
