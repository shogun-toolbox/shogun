/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Kyle McQuisten
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/lib/common.h>
#include <shogun/lib/config.h>

using namespace shogun;

CPolyKernel::CPolyKernel() : CDotKernel(0)
{
	init();
}

CPolyKernel::CPolyKernel(int32_t size, int32_t d, float64_t c, float64_t gamma)
    : CDotKernel(size)
{
	REQUIRE(c >= 0.0, "c parameter must be positive!");
	init();
	degree = d;
	m_c = c;
	m_gamma = gamma;
}

CPolyKernel::CPolyKernel(
    CDotFeatures* l, CDotFeatures* r, int32_t d, float64_t c, float64_t gamma,
    int32_t size)
    : CDotKernel(size)
{
	REQUIRE(c >= 0.0, "c parameter must be positive!");
	init();

	degree = d;
	m_c = c;
	init(l, r);
	m_gamma = gamma;
}

CPolyKernel::~CPolyKernel()
{
	cleanup();
}

bool CPolyKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);
	initialise_auto_params();
	return init_normalizer();
}

void CPolyKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CPolyKernel::compute(int32_t idx_a, int32_t idx_b)
{
	auto result = m_gamma * CDotKernel::compute(idx_a, idx_b) + m_c;
	return CMath::pow(result, degree);
}

void CPolyKernel::init()
{
	degree = 0;
	m_c = 0.0;
	m_gamma = 1.0;

	set_normalizer(new CSqrtDiagKernelNormalizer());
	SG_ADD(
	    &degree, "degree", "Degree of polynomial kernel",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_c, "c", "The kernel is inhomogeneous if the value is higher than 0",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_gamma, "gamma", "Scaler for the dot product",
	    ParameterProperties::HYPER | ParameterProperties::AUTO,
	    [num_lhs = &num_lhs]() {
		    return make_any(1.0 / static_cast<double>(*num_lhs));
	    });
}
