/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/config.h>
#include <lib/common.h>
#include <io/SGIO.h>

#include <base/Parameter.h>
#include <kernel/DiagKernel.h>

using namespace shogun;

CDiagKernel::CDiagKernel()
: CKernel()
{
	init();
}

CDiagKernel::CDiagKernel(int32_t size, float64_t d)
: CKernel(size)
{
	init();
	diag=d;
}

CDiagKernel::CDiagKernel(CFeatures* l, CFeatures* r, float64_t d)
: CKernel()
{
	init();
	diag=d;
	init(l, r);
}

CDiagKernel::~CDiagKernel()
{
}

bool CDiagKernel::init(CFeatures* l, CFeatures* r)
{
	CKernel::init(l, r);
	return init_normalizer();
}

void CDiagKernel::init()
{
	diag=1.0;
	SG_ADD(&diag, "diag", "Value on kernel diagonal.", MS_AVAILABLE);
}
