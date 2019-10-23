/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Soeren Sonnenburg, Evan Shelhamer, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/JensenShannonKernel.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

JensenShannonKernel::JensenShannonKernel()
: DotKernel(0)
{
}

JensenShannonKernel::JensenShannonKernel(int32_t size)
: DotKernel(size)
{
}

JensenShannonKernel::JensenShannonKernel(
	const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r, int32_t size)
: DotKernel(size)
{
	init(l,r);
}

JensenShannonKernel::~JensenShannonKernel()
{
	cleanup();
}

bool JensenShannonKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	bool result=DotKernel::init(l,r);
	init_normalizer();
	return result;
}

float64_t JensenShannonKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result=0;

	/* calcualte Jensen-Shannon kernel */
	for (int32_t i=0; i<alen; i++) {
		float64_t a_i = 0, b_i = 0;
		float64_t ab = avec[i]+bvec[i];
		if (avec[i] != 0)
			a_i = avec[i] * Math::log2(ab/avec[i]);
		if (bvec[i] != 0)
			b_i = bvec[i] * Math::log2(ab/bvec[i]);

		result += 0.5*(a_i + b_i);
	}

	(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->free_feature_vector(avec, idx_a, afree);
	(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

