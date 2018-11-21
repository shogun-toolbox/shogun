/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

#include <shogun/base/Parameter.h>
#include <shogun/kernel/DiagKernel.h>

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
	SG_ADD(&diag, "diag", "Value on kernel diagonal.", ParameterProperties::HYPER);
}
