/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Roman Votyakov, Evan Shelhamer, Sergey Lisitsyn, 
 *          Wu Lin
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/GaussianShiftKernel.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CGaussianShiftKernel::CGaussianShiftKernel()
: CGaussianKernel(), max_shift(0), shift_step(0)
{
	init();
}

CGaussianShiftKernel::CGaussianShiftKernel(
	int32_t size, float64_t w, int32_t ms, int32_t ss)
: CGaussianKernel(size, w), max_shift(ms), shift_step(ss)
{
	init();
}

CGaussianShiftKernel::CGaussianShiftKernel(
	CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r, float64_t w, int32_t ms, int32_t ss,
	int32_t size)
: CGaussianKernel(l, r, w, size), max_shift(ms), shift_step(ss)
{
	init();
	init(l,r);
}

CGaussianShiftKernel::~CGaussianShiftKernel()
{
}

float64_t CGaussianShiftKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result = 0.0 ;
	float64_t sum=0.0 ;
	for (int32_t i=0; i<alen; i++)
		sum+=(avec[i]-bvec[i])*(avec[i]-bvec[i]);
	result += exp(-sum/get_width()) ;

	for (int32_t shift = shift_step, s=1; shift<max_shift; shift+=shift_step, s++)
	{
		sum=0.0 ;
		for (int32_t i=0; i<alen-shift; i++)
			sum+=(avec[i+shift]-bvec[i])*(avec[i+shift]-bvec[i]);
		result += exp(-sum/get_width())/(2*s) ;

		sum=0.0 ;
		for (int32_t i=0; i<alen-shift; i++)
			sum+=(avec[i]-bvec[i+shift])*(avec[i]-bvec[i+shift]);
		result += exp(-sum/get_width())/(2*s) ;
	}

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

void CGaussianShiftKernel::init()
{
	SG_ADD(&max_shift, "max_shift", "Maximum shift.", ParameterProperties::HYPER);
	SG_ADD(&shift_step, "shift_step", "Shift stepsize.", ParameterProperties::HYPER);
}
