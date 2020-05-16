/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

#include <shogun/kernel/ConstKernel.h>
#include <shogun/features/Features.h>

#include <utility>

using namespace shogun;

ConstKernel::ConstKernel()
: Kernel()
{
	SG_ADD(&m_const_val, "const_value", "Value for kernel elements.",
	    ParameterProperties::HYPER);
}

ConstKernel::ConstKernel(float64_t c)
: ConstKernel()
{
	m_const_val = c;
}

ConstKernel::ConstKernel(std::shared_ptr<Features> l, std::shared_ptr<Features> r, float64_t c)
: ConstKernel(c)
{
	Kernel::init(l, r);
	ASSERT(init_normalizer());
}

ConstKernel::~ConstKernel()
{
}

bool ConstKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	Kernel::init(l, r);
	return init_normalizer();
}

