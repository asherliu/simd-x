#include <iostream>
#include "sorter.h"
#include <stdlib.h>

namespace trans_graph_spmv
{
	/*
	 *Show the column range or row range
	 *0-10  belongs to row 0 (means 0, ..., 9 belongs to row 0);
	 *10-9 belongs to row 1 (means nothing belongs to row 1);
	 *10~10 beongs to row 2 (means 10 belongs to row 2);
	 *The same way around
	 *------------------------------------
	 *seq can be sorted by start vertex which is row oriented;
	 *			end vertex which is column oriented;
	 *-----------------------------------
	 * */
	int ranger(edge_ptr *edge_seq, int num_of_ranges, int seq_len, 
			int **ranges, /* 0 means start index
				       * 1 means end index
				       * */
			int type)/*ranges by start vertex or end vertex*/
	{
		int start_ptr, end_ptr;
		int indicator;
		int comp_base;

		//init the range indicator
		indicator=0;
		start_ptr=0;
		end_ptr=-1;
		ranges[indicator][0]=start_ptr;
		ranges[indicator][1]=end_ptr;
		
		if(seq_len<=0)
		{
			std::cout<<"Wrong seq_len.\n";
			return -1;
		}

		for(int i=0;i<seq_len;i++)
		{
			if(type==SORT_BY_START)
			{
				comp_base=edge_seq[i]->start_vertex_id;	
			}
			else//SORT_BY_END
			{
				comp_base=edge_seq[i]->end_vertex_id;
			}

			if(comp_base==indicator)
			{
				end_ptr=i;
				ranges[indicator][1]=end_ptr;
			}else{
				indicator++;

				//fill in all the following ranges 
				//hold zero elements
				for(;indicator<comp_base;indicator++)
				{
					ranges[indicator][0]=i;
					ranges[indicator][1]=i-1;
				}
			
				start_ptr=i;
				end_ptr=i;
				ranges[indicator][0]=start_ptr;
				ranges[indicator][1]=end_ptr;
			}
		}
	
		if(indicator>=num_of_ranges)
		{
			std::cout<<"Wrong num_of_ranges.\n";
			return -2;
		}

		while(indicator<num_of_ranges-1)
		{
			indicator++;
			ranges[indicator][0]=seq_len-1;
			ranges[indicator][1]=seq_len-2;
		}
		return 0;	
	}


	/*
	 *finds the mapping from column-oriented to row-oriented
	 *based on the row_ranger
	 *---------------------------
	 *Working strategy:
	 * -Get the data to be evaluated from col_based queue
	 * -According to it's row number find the ranges in the row-oriented queue
	 * -Go through the range to find the matching item and store the index
	 * */

	int mapper(edge_ptr *col_based, edge_ptr *row_based,
			int **col_ranger, int **row_ranger,
			int seq_len, int num_ver,
			int *mapping)
	{
		int row_indicator;
		int col_value;
		int range_start, range_end;
		int *map_base_ptr=new int[num_ver];
		for(int i=0; i<num_ver; i++)
			map_base_ptr[i]=row_ranger[i][0];

		for(int i=0;i<seq_len;i++)
		{
			row_indicator	=col_based[i]->start_vertex_id;
			col_value 	=col_based[i]->end_vertex_id;
			range_start	=row_ranger[row_indicator][0];
			range_end	=row_ranger[row_indicator][1];
			
			if(range_start>range_end)
			{
				std::cout<<"ranger is wrong.\n";
				return -1;
			}
			mapping[i]=map_base_ptr[row_indicator];
			map_base_ptr[row_indicator]++;

		//	for(int j=range_start; j<=range_end;j++)
		//	{
		//		if(col_value==row_based[j]->end_vertex_id)
		//		{
		//			mapping[i]=j;
		//			break;
		//		}
		//	}
		}

		return 0;
	}

	void gen_comp_vec(std::string filename, int num_cols)
	{
		std::string comp_vec_file=filename;
		comp_vec_file.append(".comp.vec.log");
		std::ofstream myfile(comp_vec_file.c_str());
		
//		if(!myfile.is_open())
//			std::cout<<"shoot\n";
//
		//std::cout<<myfile.is_open()<<"\n";
		for(int i=0;i<num_cols;i++)
			myfile<<(rand()%234)/39.1234523<<"\n";

		myfile.close();
	}
}
