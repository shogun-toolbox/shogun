/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Björn Esser
 */

#include <shogun/base/Parameter.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/RandomFourierDotFeatures.h>

namespace shogun {

enum KernelName;

CRandomFourierDotFeatures::CRandomFourierDotFeatures()
{
	init(NOT_SPECIFIED, SGVector<float64_t>());
}

CRandomFourierDotFeatures::CRandomFourierDotFeatures(CDotFeatures* features,
	int32_t D, KernelName kernel_name, SGVector<float64_t> params)
: CRandomKitchenSinksDotFeatures(features, D)
{
	init(kernel_name, params);
	random_coeff = generate_random_coefficients();
}

CRandomFourierDotFeatures::CRandomFourierDotFeatures(CDotFeatures* features,
	int32_t D, KernelName kernel_name, SGVector<float64_t> params,
	SGMatrix<float64_t> coeff)
: CRandomKitchenSinksDotFeatures(features, D, coeff)
{
	init(kernel_name, params);
}

CRandomFourierDotFeatures::CRandomFourierDotFeatures(CFile* loader)
{
	SG_NOTIMPLEMENTED;
}

CRandomFourierDotFeatures::CRandomFourierDotFeatures(const CRandomFourierDotFeatures& orig)
: CRandomKitchenSinksDotFeatures(orig)
{
	init(orig.kernel, orig.kernel_params);
}

CRandomFourierDotFeatures::~CRandomFourierDotFeatures()
{
}

void CRandomFourierDotFeatures::init(KernelName kernel_name, SGVector<float64_t> params)
{
	kernel = kernel_name;
	kernel_params = params;

	constant = num_samples > 0 ? std::sqrt(2.0 / num_samples) : 1;
	m_parameters->add(&kernel_params, "kernel_params",
			"The parameters of the kernel to approximate");
	SG_ADD((machine_int_t* ) &kernel, "kernel",
			"The kernel to approximate", MS_NOT_AVAILABLE);
	SG_ADD(&constant, "constant", "A constant needed",
			MS_NOT_AVAILABLE);
}

CFeatures* CRandomFourierDotFeatures::duplicate() const
{
	return new CRandomFourierDotFeatures(*this);
}

const char* CRandomFourierDotFeatures::get_name() const
{
	return "RandomFourierDotFeatures";
}

float64_t CRandomFourierDotFeatures::post_dot(float64_t dot_result, index_t par_idx)
{
	dot_result += random_coeff(random_coeff.num_rows-1, par_idx);
	return std::cos(dot_result) * constant;
}

SGVector<float64_t> CRandomFourierDotFeatures::generate_random_parameter_vector()
{
	SGVector<float64_t> vec(feats->get_dim_feature_space()+1);
	switch (kernel)
	{
		case GAUSSIAN:
			for (index_t i=0; i<vec.vlen-1; i++)
			{
				vec[i] = std::sqrt((float64_t)1 / kernel_params[0]) *
				         std::sqrt(2.0) * CMath::normal_random(0.0, 1);
			}

			vec[vec.vlen-1] = CMath::random(0.0, 2 * CMath::PI);
			break;

		default:
			SG_SERROR("Unknown kernel\n");
	}
	return vec;
}

}
