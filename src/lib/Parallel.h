#ifndef PARALLEL_H__
#define PARALLEL_H__

#include "lib/common.h"

class CParallel
{
public:
	CParallel();
	~CParallel();

	static inline void set_num_threads(INT n)
	{
		num_threads=n;
	}

	static inline INT get_num_threads()
	{
		return num_threads;
	}

protected:
	static INT num_threads;
};
#endif
