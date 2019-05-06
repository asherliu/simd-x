#include <sstream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <list>
#include <vector>
#include "make_graph.h"
#include <iostream>
#include <time.h>
#include <sys/time.h>

#ifndef TIME_H
#define TIME_H
inline double get_time(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + tv.tv_usec * 1.e-6;

}
#endif

template< 	typename data_t, typename index_t,
			typename g500_t>
bool csr_vector(packed_edge		*g500_tuples,
				std::string 	csr_file,
				index_t 		edge_count,
				index_t 	vertex_count)
{
	
	double start, tm_taken;

	std::vector<std::list<data_t> > edge_vector (vertex_count);
	typename std::list<data_t>::iterator it;
	double report_criteria = 0.0;

	start = get_time();
	//CSR
	for(index_t i = 0; i< edge_count; i++){
		if((i*1.0)/edge_count >= report_criteria){
			std::cout<<(100.0*i)/edge_count<<"%\n";
			report_criteria += 0.1;
		}
		edge_vector[(data_t)(g500_tuples[i].v0)].
				push_back((data_t)(g500_tuples[i].v1));
		edge_vector[(data_t)(g500_tuples[i].v1)].
				push_back((data_t)(g500_tuples[i].v0));
	}
	tm_taken = get_time() - start;
	
	std::cout<<"================Insertion=====================\n";
	std::cout<<"Time: "<<tm_taken<<" s\n"
			<<"Rate: "<<(edge_count<<1)/tm_taken * 1.e-6<<"Medges/s\n";
	
	std::cout<<"Tuple->CSR...\n";
	report_criteria = 0.0;
//	std::ofstream myfile(csr_file.c_str());
	
	start = get_time();
	for(index_t i = 0; i< edge_vector.size(); i++)
	{
		if((i*1.0)/edge_vector.size() >= report_criteria){
			std::cout<<(100.0*i)/edge_vector.size()<<"%\n";
			report_criteria += 0.1;
		}
		edge_vector[i].sort();
//		myfile<<i<<" "<<edge_vector[i].size()<<" ";
//		for(it = edge_vector[i].begin(); it != edge_vector[i].end(); ++it)
//			myfile<<*it<<" ";
//
//		myfile<<"\n";
	}
	tm_taken = get_time() - start;

//	myfile.close();

	std::cout<<"================Sort=========================\n";
	std::cout<<"Time: "<<tm_taken<<" s\n"
			<<"Rate: "<<(edge_count<<1)/tm_taken * 1.e-6<<"Medges/s\n";
	
	return true;
}
