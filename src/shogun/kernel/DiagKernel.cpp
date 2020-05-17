/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

#include <shogun/kernel/DiagKernel.h>

#include <utility>

using namespace shogun;

DiagKernel::DiagKernel()
: Kernel()
{
	m_diag=1.0;
	SG_ADD(&m_diag, "diag", "Value on kernel diagonal.", ParameterProperties::HYPER);
}

DiagKernel::DiagKernel(int32_t size, float64_t d)
: DiagKernel()
{
	set_cache_size(size);
	m_diag = d;
}

DiagKernel::DiagKernel(std::shared_ptr<Features> l, std::shared_ptr<Features> r, float64_t d)
: DiagKernel()
{
	m_diag=d;
	ASSERT(Kernel::init(std::move(l), std::move(r)));
	init_normalizer();
}

DiagKernel::~DiagKernel()
{
}

bool DiagKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	Kernel::init(l, r);
	return init_normalizer();
}

