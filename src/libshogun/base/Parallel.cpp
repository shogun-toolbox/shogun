/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "base/Parallel.h"

CParallel::CParallel() : refcount(0), num_threads(1)
{
}

CParallel::CParallel(const CParallel& orig) : refcount(0)
{
	num_threads=orig.get_num_threads();
}

CParallel::~CParallel()
{
}
