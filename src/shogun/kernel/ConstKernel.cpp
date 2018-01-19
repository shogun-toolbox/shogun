/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

#include <shogun/base/Parameter.h>

#include <shogun/kernel/ConstKernel.h>
#include <shogun/features/Features.h>

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
