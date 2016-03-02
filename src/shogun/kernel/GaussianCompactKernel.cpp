#include <shogun/kernel/GaussianCompactKernel.h>

using namespace shogun;

CGaussianCompactKernel::CGaussianCompactKernel() : CGaussianKernel()
{
}

CGaussianCompactKernel::CGaussianCompactKernel(int32_t size, float64_t width)
                                              : CGaussianKernel(size, width)
{
}

CGaussianCompactKernel::CGaussianCompactKernel(CDotFeatures* l, CDotFeatures* r,
                                              float64_t width, int32_t size)
                                              : CGaussianKernel(l, r,
                                                                width, size)
{
}

CGaussianCompactKernel::~CGaussianCompactKernel()
{
}

float64_t CGaussianCompactKernel::compute(int32_t idx_a, int32_t idx_b)
{
    int32_t len_features, power;
    len_features=((CDotFeatures*) lhs)->get_dim_feature_space();
    power=(len_features%2==0) ? (len_features+1):len_features;

    float64_t result=distance(idx_a,idx_b);
    float64_t result_multiplier=1-(sqrt(result))/3;

    if (result_multiplier<=0)
        result_multiplier=0;
    else
        result_multiplier=pow(result_multiplier, power);

    return result_multiplier*exp(-result);
}
