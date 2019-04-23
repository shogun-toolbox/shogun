/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/WaveletKernel.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

WaveletKernel::WaveletKernel() : DotKernel(), Wdilation(0.0), Wtranslation(0.0)
{
	init();
}

WaveletKernel::WaveletKernel(int32_t size, float64_t a, float64_t c)
: DotKernel(size), Wdilation(a), Wtranslation(c)
{
	init();
}

WaveletKernel::WaveletKernel(
	std::shared_ptr<DotFeatures> l, std::shared_ptr<DotFeatures> r, int32_t size, float64_t a, float64_t c)
: DotKernel(size), Wdilation(a), Wtranslation(c)
{
	init();
	init(l,r);
}

WaveletKernel::~WaveletKernel()
{
	cleanup();
}

void WaveletKernel::cleanup()
{
}

bool WaveletKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	DotKernel::init(l, r);
	return init_normalizer();
}

void WaveletKernel::init()
{
	SG_ADD(&Wdilation, "dilation", "Dilation coefficient", ParameterProperties::HYPER);
	SG_ADD(&Wtranslation, "translation", "Translation coefficient", ParameterProperties::HYPER);
}

float64_t WaveletKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result=1;

	for (int32_t i=0; i<alen; i++)
	{
		if (Wtranslation !=0)
		{
			float64_t h1=(avec[i]-Wdilation)/Wtranslation;
			float64_t h2=(bvec[i]-Wdilation)/Wtranslation;
			float64_t res1=MotherWavelet(h1);
			float64_t res2=MotherWavelet(h2);
			result=result*res1*res2;
		}
	}

	(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->free_feature_vector(avec, idx_a, afree);
	(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
