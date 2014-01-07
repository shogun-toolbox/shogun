/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/common.h>
#include <io/SGIO.h>

#include <base/Parameter.h>

#include <kernel/ConstKernel.h>
#include <features/Features.h>

using namespace shogun;

CConstKernel::CConstKernel()
: CKernel()
{
	init();
}

CConstKernel::CConstKernel(float64_t c)
: CKernel()
{
	init();
	const_value=c;
}

CConstKernel::CConstKernel(CFeatures* l, CFeatures* r, float64_t c)
: CKernel()
{
	init();
	const_value=c;
	init(l, r);
}

CConstKernel::~CConstKernel()
{
}

bool CConstKernel::init(CFeatures* l, CFeatures* r)
{
	CKernel::init(l, r);
	return init_normalizer();
}

void CConstKernel::init()
{
	const_value=1.0;
	SG_ADD(&const_value, "const_value", "Value for kernel elements.",
	    MS_AVAILABLE);
}
