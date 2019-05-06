#include "header.h"
#include <iostream>
#include "sort_pair.hpp"

template <typename index_t,
         typename vertex_t,
         typename feature_t>
void cpu_pagerank(
        feature_t &cpu_rank_sum,
        feature_t* &dist,
        int iter_limit,
        vertex_t vert_count,
        index_t edge_count,
        index_t *beg_pos,
        vertex_t *csr){
    
    cpu_rank_sum = 0;
    feature_t *dist_prev = new feature_t[vert_count];
    dist = new feature_t[vert_count];

    feature_t init_rank = 1.0/vert_count;

    feature_t level = 0;
    feature_t rank_prev = 0;
    feature_t rank_sum = 0;
    
    for(vertex_t i = 0; i < vert_count; i ++)
    {
        dist[i] = 0; 
        if(beg_pos[i+1] - beg_pos[i] != 0)
            dist_prev[i] = init_rank /(beg_pos[i+1] - beg_pos[i]);
        else 
            dist_prev[i] = 0; 

        rank_prev += dist_prev[i];
    }
    
    std::cout<<"Initial rank sum: "<<rank_prev<<"\n";

    while(true)
    {
        for(vertex_t frontier = 0; frontier < vert_count; frontier ++)
        {
            index_t my_beg = beg_pos[frontier];
            index_t my_end = beg_pos[frontier+1];
            
            feature_t new_rank = 0;
            for(;my_beg < my_end; my_beg ++)
            {
                vertex_t nebr = csr[my_beg];
                new_rank += dist_prev[nebr];
            }
            
            if (beg_pos[frontier + 1] != beg_pos[frontier]) 
                dist[frontier] = (0.15 + 0.85 * new_rank) 
                        /(beg_pos[frontier + 1] - beg_pos[frontier]);
        }
        level ++;
    
        rank_prev = rank_sum;
        rank_sum = 0;
        for(vertex_t i = 0; i < vert_count; i++)
            //rank_sum += dist[i] * (beg_pos[i+1] - beg_pos[i]);
            rank_sum += dist[i];
            //std::cout<<dist[i] * (beg_pos[i+1] - beg_pos[i])<<"\n";
        
        std::cout<<"Iteration "<<level<<": "<<rank_sum<<"\n";
        
        if (rank_sum == rank_prev) break;
        if (level == iter_limit) break;

        for(vertex_t i = 0; i < vert_count; i++)
            if (dist[i] != dist_prev[i])
                dist_prev[i] = dist[i];

        //feature_t *tmp = dist;
        //dist = dist_prev;
        //dist_prev = tmp;
    }
    
    vertex_t *key;
    feature_t * val;
    sort_pair<feature_t, vertex_t>(dist, key, val, vert_count);
    
    std::cout<<"Top 10 ranks:\n";
    for(int i = 0; i < 10; i ++)
        std::cout<<key[i]<<": " <<val[i]<<"\n";

    cpu_rank_sum = rank_sum;

    delete[] dist_prev;
    return;
}
