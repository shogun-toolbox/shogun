/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#include "lib/common.h"
#include "kernel/DiagKernel.h"
#include "lib/io.h"

CDiagKernel::CDiagKernel(int32_t size, DREAL d)
: CKernel(size), diag(d)
{
}

CDiagKernel::CDiagKernel(CFeatures* l, CFeatures* r, DREAL d)
: CKernel(10), diag(d)
{
	init(l, r);
}

CDiagKernel::~CDiagKernel()
{
}

bool CDiagKernel::load_init(FILE* src)
{
	return false;
}

bool CDiagKernel::save_init(FILE* dest)
{
	return false;
}

