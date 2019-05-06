#include <algorithm>
#include <vector>
#include <math.h>
#include "comm.h"

template<typename T>
bool quatile( std::vector<T> data, T *array, int tries){
	array[MIN]	= data[0];
	array[FST]	= (data[15] + data[16]) * 0.5;
	array[MED]	= (data[31] + data[32]) * 0.5;
	array[TRD] 	= (data[47] + data[48]) * 0.5;
	array[MAX]	= data[63];

	array[MEN] 	= 0;
	array[STD]	= 0;
	for(int i = 0; i< tries; i++)
		array[MEN] += (data[i]/64.0);	
	
	for(int i = 0; i< tries; i++)
		array[STD] += pow((data[i] - array[MEN]), 2);
	array[STD]	= sqrt(array[STD]/tries);
	return true;
}

bool hist(	double 	*time,
			long	*edge,
			double	*teps,
			int		tries,
			double 	*st_time,
			double	*st_edge,
			double	*st_teps)
{
	std::vector<double>	dv;
	std::vector<long> 	lv;

	//time
	dv.clear();
	dv.assign(time, time + tries);
	std::stable_sort(dv.begin(), dv.end());
	quatile<double>(dv, st_time, tries);

	//edges
	dv.clear();
	lv.clear();
	lv.assign(edge, edge + tries);
	dv.assign(lv.begin(), lv.end());
	std::stable_sort(dv.begin(), dv.end());
	quatile<double>(dv, st_edge, tries);

	//teps
	dv.clear();
	dv.assign(teps, teps + tries);
	std::stable_sort(dv.begin(), dv.end());
	quatile<double>(dv, st_teps, tries);
	
	st_teps[HAM] = 0;
	st_teps[HST] = 0;
	for(int i = 0; i< tries; i++)
		st_teps[HAM] += (1.0/teps[i]);
	st_teps[HAM] = (tries * 1.0)/st_teps[HAM];
	
	for(int i = 0; i< tries; i++)
		st_teps[HST] += pow((1./teps[i] - 1./st_teps[HAM]), 2);
	st_teps[HST]	= (sqrt(st_teps[HST])/(tries -1))*st_teps[HAM]*st_teps[HAM];

	return true;
}
