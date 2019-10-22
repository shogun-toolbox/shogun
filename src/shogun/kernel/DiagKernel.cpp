/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

#include <shogun/kernel/DiagKernel.h>

using namespace shogun;

DiagKernel::DiagKernel()
: Kernel()
{
	init();
}

DiagKernel::DiagKernel(int32_t size, float64_t d)
: Kernel(size)
{
	init();
	diag=d;
}

DiagKernel::DiagKernel(std::shared_ptr<Features> l, std::shared_ptr<Features> r, float64_t d)
: Kernel()
{
	init();
	diag=d;
	init(l, r);
}

DiagKernel::~DiagKernel()
{
}

bool DiagKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	Kernel::init(l, r);
	return init_normalizer();
}

void DiagKernel::init()
{
	diag=1.0;
	SG_ADD(&diag, "diag", "Value on kernel diagonal.", ParameterProperties::HYPER);
}
