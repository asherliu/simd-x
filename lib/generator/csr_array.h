#ifndef CSR_ARRAY_H
#define CSR_ARRAY_H

#include <sstream>
#include <fstream>
#include <string>
#include <list>
#include "make_graph.h"
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <cstring>

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <inttypes.h>
#include <stdio.h>

#include "make_graph.h"

#ifndef TIME_H
#define TIME_H
inline double get_time(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1.e-6;

}
#endif

template<typename data_t, typename index_t>
void sort_csr(
	data_t	*adj_list,
	index_t	*strt_pos,
	index_t	*adj_card,
	index_t	vert_count
){
	typename std::list<data_t>::iterator it;
	std::list<data_t> mylist;
	index_t strt, card, count, i, j;
	double report_criteria	= 0.0;
	
	#ifdef _OPENMP
	#pragma omp parallel for \
	num_threads(12) \
	default(shared) \
	private(i, j, count, mylist, strt, card) \
	schedule(dynamic, 128)
	#endif
	for(i = 0; i< vert_count; i++){
		if((i*1.0)/vert_count > report_criteria){
			std::cout<<(100.0*i)/vert_count<<"%\n";
			report_criteria += 0.1;
		}
		mylist.clear();
		strt = strt_pos[i];
		card = adj_card[i];
		for(j = 0; j< card; j++){
			mylist.push_back(adj_list[strt + j]);
		}

		mylist.sort();
		count = 0;
		for(it = mylist.begin(); it != mylist.end(); ++it){
			adj_list[strt + count] = *it;
			count ++;
		}
	}
}
template<typename data_t, typename index_t>
bool csr_array( int 		argc,
				char**		argv,
			packed_edge* 	&g500_tuples,
				data_t*		&adj_list,
				index_t*	&strt_pos,
				index_t*	&adj_card,
				index_t		&edge_count,
				index_t		&vert_count,
				index_t		&scale,
				index_t		&edge_factor,
				double		&construction_time)
{
	int log_numverts;
	double start, tm_taken;
	int64_t nedges;
	double report_criteria = 0.0;
	std::cout<<"./exe log_vtx (default 16) edge_factor"
			<<" (default 16)\n";	
	
	log_numverts 		= 16; /* In base 2 */
	int64_t factor		= 16;

	if(argc >= 2) log_numverts 	= atoi(argv[1]);
	if(argc >= 3) factor		= atoi(argv[2]);
	
	scale		= log_numverts;
	edge_factor	= factor;


	std::stringstream ss;
	ss.str("");
	ss.clear();
	ss<<"kron_"<<log_numverts
		<<"_"<<edge_factor
		<<".dat";

	std::string csr_name	= ss.str();
	vert_count	= 1<<log_numverts;

	/* Start of graph generation timing */
	start = omp_get_wtime();
	make_graph(log_numverts, factor << log_numverts, \
				1, 2, &nedges, &g500_tuples);
	tm_taken = omp_get_wtime() - start;
	/* End of graph generation timing */

	edge_count = (index_t) nedges;

	std::cout<<edge_count<<" edge"<< (edge_count == 1 ? "":"s")
			<<" generated in "<<tm_taken<<"s ("
			<<1. * edge_count / tm_taken * 1.e-6
			<<" Medges/s)\n\n";
	
	adj_list			= new data_t[edge_count<<1];
	adj_card			= new index_t[vert_count];
	strt_pos			= new index_t[vert_count];
	index_t *offset		= new index_t[vert_count];
	
	memset(adj_card, 0, sizeof(index_t)*vert_count);
	memset(offset,   0, sizeof(index_t)*vert_count);
	
	#ifdef DUMP_RES
	std::ofstream file("tuple_list.dat");
	#endif
	
	for(index_t i = 0; i< edge_count; i++){
		#ifdef DUMP_RES
//		file<<g500_tuples[i].v0<<" "<<g500_tuples[i].v1<<"\n";
		#endif
		adj_card[(index_t)(g500_tuples[i].v0)] ++;
		adj_card[(index_t)(g500_tuples[i].v1)] ++;
	}

	#ifdef DUMP_RES
	file.close();
	#endif
	
	//prefix scan
	std::cout<<"Tuple->CSR...\n";
	index_t pos	= 0;
	index_t ptr_id;
	index_t i;
	report_criteria = 0.0;
	
	start = get_time();
	for(index_t i = 0; i< vert_count; i++){
		strt_pos[i] = pos;
		pos += adj_card[i];
	}
	for(i = 0; i< edge_count; i++){
		if((i*1.0)/edge_count > report_criteria){
			std::cout<<(100.0*i)/edge_count<<"%\n";
			report_criteria += 0.1;
		}
		
		ptr_id = (index_t)(g500_tuples[i].v0);
		adj_list[strt_pos[ptr_id]+offset[ptr_id]]
				= (data_t)(g500_tuples[i].v1);
		offset[ptr_id]++;

		ptr_id = (index_t)(g500_tuples[i].v1);
		adj_list[strt_pos[ptr_id]+offset[ptr_id]]
				= (data_t)(g500_tuples[i].v0);
		offset[ptr_id]++;
	}
	tm_taken = get_time() - start;
	std::cout<<"Time: "<<tm_taken<<" s\n"
		<<"Rate: "<<(edge_count<<1)/tm_taken * 1.e-6<<"Medges/s\n\n";
	construction_time	= tm_taken;

	#ifdef SORT
	std::cout<<"Sorting CSR adjacency list ... \n";
	start = get_time();
	sort_csr<data_t, index_t>
	(
		adj_list,
		strt_pos,
		adj_card,
		vert_count
	);
	tm_taken = get_time() - start;
	std::cout<<"Time: "<<tm_taken<<" s\n"
		<<"Rate: "<<(edge_count<<1)/tm_taken * 1.e-6<<"Medges/s\n\n";
	#endif

	return true;
}

#endif
