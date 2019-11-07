/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/string/SimpleLocalityImprovedStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

SimpleLocalityImprovedStringKernel::SimpleLocalityImprovedStringKernel()
: StringKernel<char>()
{
	init();
}

SimpleLocalityImprovedStringKernel::SimpleLocalityImprovedStringKernel(
	int32_t size, int32_t l, int32_t id, int32_t od)
: StringKernel<char>(size)
{
	init();

	length=l;
	inner_degree=id;
	outer_degree=od;
}

SimpleLocalityImprovedStringKernel::SimpleLocalityImprovedStringKernel(
	const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r,
	int32_t len, int32_t id, int32_t od)
: StringKernel<char>()
{
	init();

	length=len;
	inner_degree=id;
	outer_degree=od;

	init(l, r);
}

SimpleLocalityImprovedStringKernel::~SimpleLocalityImprovedStringKernel()
{
	cleanup();
}

bool SimpleLocalityImprovedStringKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	bool result = StringKernel<char>::init(l,r);

	if (!result)
		return false;
	const int32_t num_features = std::static_pointer_cast<StringFeatures<char>>(l)->get_max_vector_length();
	const int32_t PYRAL = 2 * length - 1; // total window length
	const int32_t pyra_len  = num_features-PYRAL+1;
	const int32_t pyra_len2 = (int32_t) pyra_len/2;

	pyramid_weights = SGVector<float64_t>(pyra_len);

	SG_DEBUG("initializing pyramid weights: size={} length={}",
		num_features, length);

	float64_t PYRAL_pot;
	int32_t DEGREE1_1  = (inner_degree & 0x1)==0;
	int32_t DEGREE1_1n = (inner_degree & ~0x1)!=0;
	int32_t DEGREE1_2  = (inner_degree & 0x2)!=0;
	int32_t DEGREE1_3  = (inner_degree & ~0x3)!=0;
	int32_t DEGREE1_4  = (inner_degree & 0x4)!=0;
	{
	float64_t PYRAL_ = PYRAL;
	PYRAL_pot = DEGREE1_1 ? 1.0 : PYRAL_;
	if (DEGREE1_1n)
	{
		PYRAL_ *= PYRAL_;
		if (DEGREE1_2)
			PYRAL_pot *= PYRAL_;
		if (DEGREE1_3)
		{
			PYRAL_ *= PYRAL_;
			if (DEGREE1_4)
				PYRAL_pot *= PYRAL_;
		}
	}
	}

	{
	int32_t j;
	for (j = 0; j < pyra_len; j++)
		pyramid_weights[j] = 4*((float64_t)((j < pyra_len2)? j+1 : pyra_len-j))/((float64_t)pyra_len);
	for (j = 0; j < pyra_len; j++)
		pyramid_weights[j] /= PYRAL_pot;
	}

	return init_normalizer();
}

void SimpleLocalityImprovedStringKernel::cleanup()
{
	pyramid_weights = SGVector<float64_t>();
	Kernel::cleanup();
}

float64_t SimpleLocalityImprovedStringKernel::dot_pyr (const char* const x1,
	     const char* const x2, const int32_t NOF_NTS, const int32_t NTWIDTH,
	     const int32_t DEGREE1, const int32_t DEGREE2, float64_t *pyra)
{
	const int32_t PYRAL = 2*NTWIDTH-1; // total window length
	float64_t pot;
	float64_t sum;
	int32_t DEGREE1_1 = (DEGREE1 & 0x1)==0;
	int32_t DEGREE1_1n = (DEGREE1 & ~0x1)!=0;
	int32_t DEGREE1_2 = (DEGREE1 & 0x2)!=0;
	int32_t DEGREE1_3 = (DEGREE1 & ~0x3)!=0;
	int32_t DEGREE1_4 = (DEGREE1 & 0x4)!=0;

	ASSERT((DEGREE1 & ~0x7) == 0)
	ASSERT((DEGREE2 & ~0x7) == 0)

	int32_t conv;
	int32_t i;
	int32_t j;

	sum = 0.0;
	conv = 0;
	for (j = 0; j < PYRAL; j++)
		conv += (x1[j] == x2[j]) ? 1 : 0;

	for (i = 0; i < NOF_NTS-PYRAL+1; i++)
	{
		float64_t pot2;
		if (i>0)
			conv += ((x1[i+PYRAL-1] == x2[i+PYRAL-1]) ? 1 : 0 ) -
				((x1[i-1] == x2[i-1]) ? 1 : 0);
		{ /* potencing of conv -- float64_t is faster*/
		float64_t conv2 = conv;
		pot2 = (DEGREE1_1) ? 1.0 : conv2;
			if (DEGREE1_1n)
			{
				conv2 *= conv2;
				if (DEGREE1_2)
					pot2 *= conv2;
				if (DEGREE1_3 && DEGREE1_4)
					pot2 *= conv2*conv2;
			}
		}
		sum += pot2*pyra[i];
	}

	pot = ((DEGREE2 & 0x1) == 0) ? 1.0 : sum;
	if ((DEGREE2 & ~0x1) != 0)
	{
		sum *= sum;
		if ((DEGREE2 & 0x2) != 0)
			pot *= sum;
		if ((DEGREE2 & ~0x3) != 0)
		{
			sum *= sum;
			if ((DEGREE2 & 0x4) != 0)
				pot *= sum;
		}
	}
	return pot;
}

float64_t SimpleLocalityImprovedStringKernel::compute(
	int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	char* avec = std::static_pointer_cast<StringFeatures<char>>(lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = std::static_pointer_cast<StringFeatures<char>>(rhs)->get_feature_vector(idx_b, blen, free_bvec);

	// can only deal with strings of same length
	ASSERT(alen==blen)

	float64_t dpt;

	dpt = dot_pyr(avec, bvec, alen, length, inner_degree, outer_degree, pyramid_weights);
	dpt = dpt / pow((float64_t) alen, (float64_t) outer_degree);

	std::static_pointer_cast<StringFeatures<char>>(lhs)->free_feature_vector(avec, idx_a, free_avec);
	std::static_pointer_cast<StringFeatures<char>>(rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return (float64_t) dpt;
}

void SimpleLocalityImprovedStringKernel::init()
{
	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>());

	length = 3;
	inner_degree = 3;
	outer_degree = 1;

	SG_ADD(&length, "length", "Window Length.", ParameterProperties::HYPER);
	SG_ADD(&inner_degree, "inner_degree", "Inner degree.", ParameterProperties::HYPER);
	SG_ADD(&outer_degree, "outer_degree", "Outer degree.", ParameterProperties::HYPER);
	SG_ADD(&pyramid_weights,"pyramid_weights", "Pyramid weights.", ParameterProperties::HYPER);
}
