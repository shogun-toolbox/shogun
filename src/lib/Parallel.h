/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

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
