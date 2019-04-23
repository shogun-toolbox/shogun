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
	init();
}

SigmoidKernel::SigmoidKernel(int32_t size, float64_t g, float64_t c)
    : DotKernel(size)
{
	init();

	gamma = g;
	coef0 = c;
}

SigmoidKernel::SigmoidKernel(
    std::shared_ptr<DotFeatures> l, std::shared_ptr<DotFeatures> r, int32_t size, float64_t g, float64_t c)
    : DotKernel(size)
{
	init();

	gamma = g;
	coef0 = c;

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

void SigmoidKernel::init()
{
	gamma = 0.0;
	coef0 = 0.0;

	SG_ADD(
	    &gamma, "gamma", "Scaler for the dot product.",
	    ParameterProperties::HYPER | ParameterProperties::AUTO,
	    std::make_shared<params::GammaFeatureNumberInit>(this));
	SG_ADD(&coef0, "coef0", "Coefficient 0.", ParameterProperties::HYPER);
}
