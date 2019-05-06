#include "header.h"
#include <iostream>
#include "sort_pair.hpp"

template <typename index_t,
         typename vertex_t,
         typename feature_t>
void cpu_delta_pagerank(
        feature_t* &dist,
        vertex_t vert_count,
        index_t edge_count,
        index_t *beg_pos,
        vertex_t *csr){
    
    feature_t *dist_delta = new feature_t[vert_count];
    feature_t *dist_delta_next = new feature_t[vert_count];
    dist = new feature_t[vert_count];
    feature_t *degree_reverse = new feature_t[vert_count];

    feature_t init_rank = 1.0/vert_count;
    
    //Compute init pagerank value 1/vert_count
    //We multiply init_rank * degree_reverse 
    //- to save the division operation.
    std::cout<<init_rank<<"\n";
    for(vertex_t i = 0; i < vert_count; i ++)
    {
        if(beg_pos[i+1] - beg_pos[i] != 0)
            degree_reverse[i] = 1.0/(beg_pos[i+1] - beg_pos[i]);
        else degree_reverse[i] = 0;

        dist[i] = init_rank * degree_reverse[i];
    }
     
    feature_t level = 0;
    //Compute init delta that will pass around in the graph
    for(vertex_t i = 0; i < vert_count; i ++)
    {
        vertex_t frontier = i;

        index_t my_beg = beg_pos[frontier];
        index_t my_end = beg_pos[frontier+1];

        feature_t new_rank = 0;
        for(;my_beg < my_end; my_beg ++)
        {
            vertex_t nebr = csr[my_beg];
            new_rank += dist[nebr];
        }
    
        new_rank = (0.15 + 0.85 * new_rank) * degree_reverse[frontier];
        
        //Init delta update
        dist_delta[frontier] = new_rank - dist[frontier];
        dist_delta_next[frontier] = new_rank;//use dist_delta_next as a temp store for new rank
        
    }

    //put new rank back to array dist
    for(vertex_t i = 0; i < vert_count; i ++)
        dist[i] = dist_delta_next[i];

    feature_t rank_sum = 0;
    for(vertex_t i = 0; i < vert_count; i++)
        rank_sum += dist[i] * (beg_pos[i+1] - beg_pos[i]);

    std::cout<<"Iteration "<<(level++)<<": "<<rank_sum<<"\n";
    
    //Converging on delta
    feature_t rank_prev = 0;

    while(true)
    {
        index_t update_count = 0;
        for(vertex_t i = 0; i < vert_count; i ++)
        {
            vertex_t frontier = i;
            index_t my_beg = beg_pos[frontier];
            index_t my_end = beg_pos[frontier+1];
            
            feature_t new_rank = 0;
            for(;my_beg < my_end; my_beg ++)
            {
                vertex_t nebr = csr[my_beg];
                //if(dist_delta[nebr] >=0.0001) 
                if(dist_delta[nebr] > 0) 
                {
                    update_count ++;
                    new_rank += dist_delta[nebr];
                }
            }
            dist_delta_next[frontier] = 0.85 * new_rank * degree_reverse[frontier];
        }
        level ++;
        
        rank_prev = rank_sum;
        rank_sum = 0;
        for(vertex_t i = 0; i < vert_count; i++)
        {
            dist[i] += dist_delta_next[i];
            rank_sum += dist[i] * (beg_pos[i+1] - beg_pos[i]);
        }

        std::cout<<"Iteration "<<level<<": "<<rank_sum<<" update ratio: "<<update_count * 1.0/edge_count<<"\n";
        
        if(rank_prev == rank_sum) break;
        if(update_count == 0) break;
        feature_t *tmp = dist_delta;
        dist_delta = dist_delta_next;
        dist_delta_next = tmp;
    }
    
    vertex_t *key;
    feature_t * val;
    sort_pair<feature_t, vertex_t>(dist, key, val, vert_count);
    
    std::cout<<"Top 10 ranks:\n";
    for(int i = 0; i < 10; i ++)
        std::cout<<key[i]<<": " <<val[i]<<"\n";

    delete[] dist_delta;
    delete[] dist_delta_next;
    delete[] degree_reverse;
    return;
}
