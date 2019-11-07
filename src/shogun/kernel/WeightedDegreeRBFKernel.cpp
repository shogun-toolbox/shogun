/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Sergey Lisitsyn,
 *          Leon Kuchenbecker
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/WeightedDegreeRBFKernel.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

WeightedDegreeRBFKernel::WeightedDegreeRBFKernel()
: DotKernel(), width(1), degree(1), weights(0)
{
	register_params();
}


WeightedDegreeRBFKernel::WeightedDegreeRBFKernel(int32_t size, float64_t w, int32_t d, int32_t nof_prop)
: DotKernel(size), width(w), degree(d), nof_properties(nof_prop), weights(0)
{
	init_wd_weights();
	register_params();
}

WeightedDegreeRBFKernel::WeightedDegreeRBFKernel(
	const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r, float64_t w, int32_t d, int32_t nof_prop, int32_t size)
: DotKernel(size), width(w), degree(d), nof_properties(nof_prop), weights(0)
{
	init_wd_weights();
	register_params();
	init(l,r);
}

WeightedDegreeRBFKernel::~WeightedDegreeRBFKernel()
{
	SG_FREE(weights);
	weights=NULL;
}

bool WeightedDegreeRBFKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	DotKernel::init(l, r);
	SG_DEBUG("Initialized WeightedDegreeRBFKernel ({}).", fmt::ptr(this))
	return init_normalizer();
}

bool WeightedDegreeRBFKernel::init_wd_weights()
{
	ASSERT(degree>0)

	if (weights!=0)	SG_FREE(weights);
	weights=SG_MALLOC(float64_t, degree);
	if (weights)
	{
		int32_t i;
		float64_t sum=0;
		for (i=0; i<degree; i++)
		{
			weights[i]=degree-i;
			sum+=weights[i];
		}
		for (i=0; i<degree; i++)
			weights[i]/=sum;

		SG_DEBUG("Initialized weights for WeightedDegreeRBFKernel ({}).", fmt::ptr(this))
		return true;
	}
	else
		return false;
}


float64_t WeightedDegreeRBFKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)
	ASSERT(alen%nof_properties == 0)

	float64_t result=0;

	for (int32_t i=0; i<alen; i+=nof_properties)
	{
		float64_t resulti = 0.0;

		for (int32_t d=0; (i+(d*nof_properties)<alen) && (d<degree); d++)
		{
			float64_t resultid = 0.0;
			int32_t limit = (d + 1 ) * nof_properties;
			for (int32_t k=0; k < limit; k++)
			{
				resultid+=Math::sq(avec[i+k]-bvec[i+k]);
			}

			resulti += weights[d] * exp(-resultid/width);
		}

		result+=resulti ;
	}

	return result;
}

void WeightedDegreeRBFKernel::register_params()
{
	SG_ADD(&width, "width", "Kernel width", ParameterProperties::HYPER);
	SG_ADD(&degree, "degree", "Kernel degree", ParameterProperties::HYPER);
}
