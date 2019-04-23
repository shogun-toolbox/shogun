#include <shogun/kernel/GaussianCompactKernel.h>
#include <shogun/features/DotFeatures.h>

using namespace shogun;

GaussianCompactKernel::GaussianCompactKernel() : GaussianKernel()
{
}

GaussianCompactKernel::GaussianCompactKernel(int32_t size, float64_t width)
                                              : GaussianKernel(size, width)
{
}

GaussianCompactKernel::GaussianCompactKernel(std::shared_ptr<DotFeatures> l, std::shared_ptr<DotFeatures> r,
                                              float64_t width, int32_t size)
                                              : GaussianKernel(l, r,
                                                                width, size)
{
}

GaussianCompactKernel::~GaussianCompactKernel()
{
}

float64_t GaussianCompactKernel::compute(int32_t idx_a, int32_t idx_b)
{
    int32_t len_features, power;
    len_features=(std::static_pointer_cast<DotFeatures>(lhs))->get_dim_feature_space();
    power=(len_features%2==0) ? (len_features+1):len_features;

    float64_t result=distance(idx_a,idx_b);
	float64_t result_multiplier = 1 - (std::sqrt(result)) / 3;

	if (result_multiplier <= 0)
		return 0;

	return Math::pow(result_multiplier, power) * std::exp(-result);
}
