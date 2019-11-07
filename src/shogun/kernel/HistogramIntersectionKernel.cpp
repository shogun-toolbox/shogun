/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Sergey Lisitsyn, Viktor Gal
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/HistogramIntersectionKernel.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

HistogramIntersectionKernel::HistogramIntersectionKernel()
: DotKernel(0), m_beta(1.0)
{
	register_params();
}

HistogramIntersectionKernel::HistogramIntersectionKernel(int32_t size)
: DotKernel(size), m_beta(1.0)
{
	register_params();
}

HistogramIntersectionKernel::HistogramIntersectionKernel(
	const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r,
	float64_t beta, int32_t size)
: DotKernel(size), m_beta(beta)
{
	init(l,r);
	register_params();
}

HistogramIntersectionKernel::~HistogramIntersectionKernel()
{
	cleanup();
}

bool HistogramIntersectionKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	bool result=DotKernel::init(l,r);
	init_normalizer();
	return result;
}

float64_t HistogramIntersectionKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result=0;

	// checking if beta is default or not
	if (m_beta == 1.0)
	{
		// compute standard histogram intersection kernel
		for (int32_t i=0; i<alen; i++)
			result += (avec[i] < bvec[i]) ? avec[i] : bvec[i];
	}
	else
	{
		//compute generalized histogram intersection kernel
		for (int32_t i=0; i<alen; i++)
			result += Math::min(Math::pow(avec[i],m_beta), Math::pow(bvec[i],m_beta));
	}
	(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->free_feature_vector(avec, idx_a, afree);
	(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

void HistogramIntersectionKernel::register_params()
{
	SG_ADD(&m_beta, "beta", "the beta parameter of the kernel", ParameterProperties::HYPER);
}
