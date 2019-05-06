#include "header.h"
#include "util.h"

template <typename index_t,
         typename vertex_t,
         typename weight_t,
         typename feature_t>
void cpu_sssp(
        feature_t* &dist,
        vertex_t src,
        vertex_t vert_count,
        index_t edge_count,
        index_t *beg_pos,
        vertex_t *csr,
        weight_t *weight){
    
    dist = new feature_t[vert_count];
    for(vertex_t i = 0; i < vert_count; i ++)
        dist[i] = INFTY;

    vertex_t *fq = new vertex_t[edge_count];
    vertex_t *fq_next = new vertex_t[edge_count];

    vertex_t fq_count, fq_next_count;

    fq[0] = src;
    dist[src] = 0;
    fq_count = 1;
    feature_t level = 0;
    while(true)
    {
        fq_next_count = 0;
        for(vertex_t i = 0; i < fq_count; i ++)
        {
            vertex_t frontier = fq[i];
            feature_t frontier_dist = dist[frontier];

            index_t my_beg = beg_pos[frontier];
            index_t my_end = beg_pos[frontier+1];
      
            //std::cout<<my_beg<<" "<<my_end<<"\n";
            for(;my_beg < my_end; my_beg ++)
            {
                vertex_t nebr = csr[my_beg];
                weight_t weit = weight[my_beg];
                //if (nebr == 0) std::cout<<frontier_dist<<" + "<< weit<<" = "<< weit+frontier_dist<<"\n"; 
                if(dist[nebr] > weit + frontier_dist)
                {
                    dist[nebr] = weit + frontier_dist;
                    fq_next[fq_next_count ++] = nebr;
                }
            }
        }
        
        if (fq_next_count == 0) break;
        fq_count = fq_next_count;
        vertex_t *tmp = fq;
        fq = fq_next;
        fq_next = tmp;

        //std::cout<<"Level-"<<(int)level<<" Frontier count: "<<fq_count<<"\n";
        level ++;
    }
    
    delete[] fq;
    delete[] fq_next;
    return;
}
