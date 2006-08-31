/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "kernel/ConstKernel.h"
#include "features/Features.h"
#include "lib/io.h"

CConstKernel::CConstKernel(DREAL c) : CKernel(0), const_value(c)
{
}

CConstKernel::~CConstKernel() 
{
}

bool CConstKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CKernel::init(l, r, do_init); 
	return true;
}

bool CConstKernel::load_init(FILE* src)
{
	return false;
}

bool CConstKernel::save_init(FILE* dest)
{
	return false;
}
