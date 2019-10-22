/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

#include <shogun/kernel/ConstKernel.h>
#include <shogun/features/Features.h>

using namespace shogun;

ConstKernel::ConstKernel()
: Kernel()
{
	init();
}

ConstKernel::ConstKernel(float64_t c)
: Kernel()
{
	init();
	const_value=c;
}

ConstKernel::ConstKernel(std::shared_ptr<Features> l, std::shared_ptr<Features> r, float64_t c)
: Kernel()
{
	init();
	const_value=c;
	init(l, r);
}

ConstKernel::~ConstKernel()
{
}

bool ConstKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	Kernel::init(l, r);
	return init_normalizer();
}

void ConstKernel::init()
{
	const_value=1.0;
	SG_ADD(&const_value, "const_value", "Value for kernel elements.",
	    ParameterProperties::HYPER);
}
