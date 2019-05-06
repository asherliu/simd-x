#include "header.h"
#include <iostream>

template <typename index_t,
         typename vertex_t,
         typename weight_t,
         typename feature_t>
void cpu_kcore(
        index_t &kcore_count,
        feature_t* &dist,
        feature_t k,
        vertex_t vert_count,
        index_t edge_count,
        index_t *beg_pos,
        vertex_t *csr){
    
    kcore_count = 0;

    dist = new feature_t[vert_count];
    feature_t *dist_prev = new feature_t[vert_count];
    index_t active_count = 0;
    for(vertex_t i = 0; i < vert_count; i ++)
    {
        dist[i] = beg_pos[i+1] - beg_pos[i];
        dist_prev[i] = beg_pos[i+1] - beg_pos[i];
    }

    index_t update_count;
    index_t total_count;
    //THIS WILL REPEATEDLY UPDATE THE SAME >K VERTEX!!!!! WRONG!
    //Two dist array can fix the problem.
    //Or only use pull once@
    for(vertex_t i = 0; i < vert_count; i ++)
    {
        //THIS WILL REPEATEDLY UPDATE THE SAME >K VERTEX!!!!! WRONG!
        //Two dist array can fix the problem.
        //Or only use pull once@
        if( dist_prev[i] > k )
        {
            vertex_t frontier = i;
            //dist[frontier] = 0; //deactivate myself

            index_t my_beg = beg_pos[frontier];
            index_t my_end = beg_pos[frontier+1];
            total_count += my_end - my_beg;

            for(;my_beg < my_end; my_beg ++)
            {
                vertex_t nebr = csr[my_beg];
                if(dist_prev[nebr] <= k && dist[frontier] > k)
                {
                    --dist[frontier];
                    ++update_count;
                }
            }
        }
    }
    
    //deactive those already <= k vertices.
    for(vertex_t i = 0; i < vert_count; i++)
        if(dist_prev[i] <= k) dist[i] = 0;

    feature_t level = 0;
    while(true)
    {
        update_count = 0;
        total_count = 0;
        active_count = 0;
        
        for(vertex_t i = 0; i < vert_count; i ++)
            active_count += ((dist[i] <= k) && (dist[i] != 0));

        for(vertex_t i = 0; i < vert_count; i ++)
        {
            if( (dist[i] <= k) && (dist[i] != 0) )
            {
                vertex_t frontier = i;
                //dist[frontier] = -1 * vert_count; //deactivate myself
                dist[frontier] = 0; //deactivate myself

                index_t my_beg = beg_pos[frontier];
                index_t my_end = beg_pos[frontier+1];
                total_count += my_end - my_beg;

                for(;my_beg < my_end; my_beg ++)
                {
                    vertex_t nebr = csr[my_beg];
                    if(dist[nebr] > k)
                    {
                        --dist[nebr];
                        ++update_count;
                    }
                    
                }
            }
        }
        
        //std::cout<<"Active count: "<< active_count<<"\n";
        level ++;
        //std::cout<<"Iteration "<<level<<", total work ratio, pratically executed ratio "
        //    <<total_count*1.0/edge_count<<" "<<update_count*1.0/edge_count<<"\n";
        //break;        
        if (update_count == 0) break;
    }

    for(vertex_t i = 0; i < vert_count; i ++)
        if(dist[i] > k ) kcore_count ++;
        //else assert(dist[i] == -1 * vert_count);
    
    std::cout<<k<<"-core vertex count: "<<kcore_count<<"\n";

    delete[] dist_prev;
    return;
}
