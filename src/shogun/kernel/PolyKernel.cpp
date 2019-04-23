/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Kyle McQuisten
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/lib/auto_initialiser.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>

using namespace shogun;

PolyKernel::PolyKernel() : DotKernel(0)
{
	init();
}

PolyKernel::PolyKernel(int32_t size, int32_t d, float64_t c, float64_t gamma)
    : DotKernel(size)
{
	require(c >= 0.0, "c parameter must be positive!");
	init();
	degree = d;
	m_c = c;
	m_gamma = gamma;
}

PolyKernel::PolyKernel(
    std::shared_ptr<DotFeatures> l, std::shared_ptr<DotFeatures> r, int32_t d, float64_t c, float64_t gamma,
    int32_t size)
    : DotKernel(size)
{
	require(c >= 0.0, "c parameter must be positive!");
	init();

	degree = d;
	m_c = c;
	init(l, r);
	m_gamma = gamma;
}

PolyKernel::~PolyKernel()
{
	cleanup();
}

bool PolyKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	DotKernel::init(l, r);
	return init_normalizer();
}

void PolyKernel::cleanup()
{
	Kernel::cleanup();
}

float64_t PolyKernel::compute(int32_t idx_a, int32_t idx_b)
{
	auto result = m_gamma * DotKernel::compute(idx_a, idx_b) + m_c;
	return Math::pow(result, degree);
}

void PolyKernel::init()
{
	degree = 0;
	m_c = 0.0;
	m_gamma = 1.0;
	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>());
	SG_ADD(
	    &degree, "degree", "Degree of polynomial kernel",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_c, "c", "The kernel is inhomogeneous if the value is higher than 0",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_gamma, "gamma", "Scaler for the dot product",
	    ParameterProperties::HYPER | ParameterProperties::AUTO,
	    std::make_shared<params::GammaFeatureNumberInit>(this));
}
