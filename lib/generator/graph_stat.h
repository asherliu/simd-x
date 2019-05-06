#include "sorter.h"
#include <string>
struct graph_stat{
	static int degree_statistic(int *counter, int num_ver, std::string filename)
	{//number of vertices has degrees at 0
		//number of vertices has degrees at 1 ....
		//
		int *stat=new int[num_ver];

		std::cout<<"here\n";
		for(int i=0;i<num_ver;i++)
			stat[i]=0;

		for(int i=0;i<num_ver;i++)
		{
			if(counter[i] > num_ver) return -1;//means one vertex out-degree
												//is larger then the total vertices
			stat[counter[i]]++;
		}
		
		filename.append(".degree_density.log");
		std::ofstream myfile(filename.c_str());

		myfile<<"Number of vertices at each degree\n";
		myfile<<"Degree\tNum of vertices\n";
		for(int i=0;i<num_ver;i++)
			myfile<<i<<"\t"<<stat[i]<<"\n";

		return 0;
	}


	static int threshold_statistic(int *counter, 
							int num_ver, 
							int num_edges, 
							int threshold, 
							std::string filename)
	{
		int i;
		int th_counter=0;//number of vertices has more than threshold counter degree
		int workload_lg_th=0;//total workload held by the vertices that has 
								//degrees larger than threshold
		int workload_ls_th=0;//total workload held by the vertices that has 
								//degrees smaller than threshold

		for(i=0;i<num_ver;i++)
		{
			if(counter[i]>threshold)
			{
				th_counter++;
				workload_lg_th+=counter[i];
			}else{
				workload_ls_th+=counter[i];	
			}
		}
		
		filename.append(".stat_conclusion.log");
		std::ofstream myfile(filename.c_str());
		myfile<<"VER_COUNT (Deg > "<<threshold<<"):\t"<<th_counter<<"\n";
		myfile<<"VER_COUNT (Deg <="<<threshold<<"):\t"<<num_ver-th_counter<<"\n";
		myfile<<"----------------------------------------------\n";
		myfile<<"WORKLOAD by Deg > "<<threshold<<":\t"<<workload_lg_th<<"\n";
		myfile<<"RATIO:"<<workload_lg_th/(num_edges*1.0)<<"\n";
		myfile.close();
		return 0;
	}

	static int ver_degree(
			edge_ptr sorted_edges,//sorted edgelists produced by sorter
			int num_ver, //number of vertices in this edgelists
			int num_edges,
			int *count,//output of each vertex out-degree
			std::string filename)
	{
		int temp_id;
		int set_id;
		int temp_counter;
	

		//init all counts
		for(int i = 0;i<num_ver;i++)
			count[i] = 0;

		int seg_id	= sorted_edges[0].start_vertex_id;
		temp_id		= seg_id;
		temp_counter= 1;
		for(int i = 1; i < num_edges;i++)
		{
			temp_id = sorted_edges[i].start_vertex_id;
			if(temp_id == seg_id)
			{
				temp_counter++;
			}else{
				//update the previous one
				count[seg_id]	= temp_counter;
				seg_id			= temp_id;
				temp_counter	= 1;
			}
		}

		//+---------------------------------
		//|LAST ONE IS NOT RECORDED YET
		//+---------------------------------
		count[seg_id]=temp_counter;	
		
		filename.append(".ver_degree.log");
		std::ofstream ofile(filename.c_str());
		for(int i=0;i<num_ver;i++)
			ofile<<"ver "<<i<<":\t"<<count[i]<<"\n";

		ofile.close();
		return 0;
	}
};
