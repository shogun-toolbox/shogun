/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Heiko Strathmann, Saurabh Goyal,
 *          Sergey Lisitsyn
 */

#include <shogun/mathematics/Math.h>
#include <shogun/kernel/ANOVAKernel.h>

using namespace shogun;

ANOVAKernel::ANOVAKernel(): DotKernel(0)
{
	SG_ADD(
	    &m_cardinality, "cardinality", "Kernel cardinality.",
	    ParameterProperties::HYPER);
}

ANOVAKernel::ANOVAKernel(int32_t size, int32_t d)
: ANOVAKernel()
{
	set_cache_size(size);
	m_cardinality = d;
}

ANOVAKernel::ANOVAKernel(
	const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r, int32_t d, int32_t cache)
  : ANOVAKernel(cache, d)
{
	init(l, r);
}

ANOVAKernel::~ANOVAKernel()
{
	cleanup();
}

bool ANOVAKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	cleanup();

	bool result = DotKernel::init(l,r);

	init_normalizer();
	return result;
}

float64_t ANOVAKernel::compute(int32_t idx_a, int32_t idx_b)
{
	return compute_rec1(idx_a, idx_b);
}

float64_t ANOVAKernel::compute_rec1(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result = compute_recursive1(avec, bvec, alen);

	(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->free_feature_vector(avec, idx_a, afree);
	(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

float64_t ANOVAKernel::compute_rec2(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result = compute_recursive2(avec, bvec, alen);

	(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->free_feature_vector(avec, idx_a, afree);
	(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

float64_t ANOVAKernel::compute_recursive1(float64_t* avec, float64_t* bvec, int32_t len)
{
	int32_t DP_len=(m_cardinality+1)*(len+1);
	float64_t* DP = SG_MALLOC(float64_t, DP_len);

	ASSERT(DP)
	int32_t d=m_cardinality;
	int32_t offs=m_cardinality+1;

	ASSERT(DP_len==(len+1)*offs)

	for (int32_t j=0; j < len+1; j++)
		DP[j] = 1.0;

	for (int32_t k=1; k < d+1; k++)
	{
		// TRAP d>len case
		if (k-1>=len)
			return 0.0;

		DP[k*offs+k-1] = 0;
		for (int32_t j=k; j < len+1; j++)
			DP[k*offs+j]=DP[k*offs+j-1]+avec[j-1]*bvec[j-1]*DP[(k-1)*offs+j-1];
	}

	float64_t result=DP[d*offs+len];

	SG_FREE(DP);

	return result;
}

float64_t ANOVAKernel::compute_recursive2(float64_t* avec, float64_t* bvec, int32_t len)
{
	float64_t* KD = SG_MALLOC(float64_t, m_cardinality+1);
	float64_t* KS = SG_MALLOC(float64_t, m_cardinality+1);
	float64_t* vec_pow = SG_MALLOC(float64_t, len);

	ASSERT(vec_pow)
	ASSERT(KS)
	ASSERT(KD)

	int32_t d=m_cardinality;
	for (int32_t i=0; i < len; i++)
		vec_pow[i] = 1;

	for (int32_t k=1; k < d+1; k++)
	{
		KS[k] = 0;
		for (int32_t i=0; i < len; i++)
		{
			vec_pow[i] *= avec[i]*bvec[i];
			KS[k] += vec_pow[i];
		}
	}

	KD[0] = 1;
	for (int32_t k=1; k < d+1; k++)
	{
		float64_t sum = 0;
		for (int32_t s=1; s < k+1; s++)
		{
			float64_t sign = 1.0;
			if (s % 2 == 0)
				sign = -1.0;

			sum += sign*KD[k-s]*KS[s];
		}

		KD[k] = sum / k;
	}
	float64_t result=KD[d];
	SG_FREE(vec_pow);
	SG_FREE(KS);
	SG_FREE(KD);

	return result;
}