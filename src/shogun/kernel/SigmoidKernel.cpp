/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/SigmoidKernel.h>
#include <shogun/lib/auto_initialiser.h>
#include <shogun/lib/common.h>

using namespace shogun;

SigmoidKernel::SigmoidKernel() : DotKernel()
{
	SG_ADD(
	    &m_gamma, "gamma", "Scaler for the dot product.",
	    ParameterProperties::HYPER | ParameterProperties::AUTO,
	    std::make_shared<params::GammaFeatureNumberInit<SigmoidKernel>>(*this, 0.0));
	SG_ADD(&coef0, "coef0", "Coefficient 0.", ParameterProperties::HYPER);
}

SigmoidKernel::SigmoidKernel(int32_t size, float64_t g, float64_t c)
    : SigmoidKernel()
{
	set_cache_size(size);
	m_gamma = g;
	coef0 = c;
}

SigmoidKernel::SigmoidKernel(
    const std::shared_ptr<DotFeatures>& l, const std::shared_ptr<DotFeatures>& r, int32_t size, float64_t g, float64_t c)
    : SigmoidKernel(size, g, c)
{
	init(l, r);
}

SigmoidKernel::~SigmoidKernel()
{
	cleanup();
}

void SigmoidKernel::cleanup()
{
}

bool SigmoidKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	DotKernel::init(l, r);
	return init_normalizer();
}
