#ifndef __SORT_PAIR_H__
#define __SORT_PAIR_H__

#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <vector>
#include "wtime.h"

struct data
{
    int key;
    double val;
};

int compare(data d1, data d2)
{
    return d1.val > d2.val;
}

template <typename val_t, typename index_t>
void sort_pair(val_t * dist, index_t* &key, val_t* &val, index_t count)
{
    data *d=new data[count];
    for(index_t i=0;i<count;++i)
    {
        d[i].key = i;
        d[i].val = dist[i];
    }

    double tm=wtime();
    std::sort(d,d+count,compare);

    std::cout<<"Takes: "<<wtime()-tm<<" seconds\n";
    std::cout<<"------------------\n\n";
    
    key = new index_t[count];
    val = new val_t[count];
    for(index_t i=0;i<count;++i)
    {
        key[i] = d[i].key;
        val[i] = d[i].val;
    }

    delete[] d;
    return;
}
#endif
