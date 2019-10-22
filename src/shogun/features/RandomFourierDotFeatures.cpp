/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Bjoern Esser
 */

#include <shogun/mathematics/Math.h>
#include <shogun/features/RandomFourierDotFeatures.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/UniformRealDistribution.h>

namespace shogun {

enum KernelName;

RandomFourierDotFeatures::RandomFourierDotFeatures()
{
	init(NOT_SPECIFIED, SGVector<float64_t>());
}

RandomFourierDotFeatures::RandomFourierDotFeatures(std::shared_ptr<DotFeatures> features,
	int32_t D, KernelName kernel_name, SGVector<float64_t> params)
: RandomKitchenSinksDotFeatures(features, D)
{
	init(kernel_name, params);
	random_coeff = generate_random_coefficients();
}

RandomFourierDotFeatures::RandomFourierDotFeatures(std::shared_ptr<DotFeatures> features,
	int32_t D, KernelName kernel_name, SGVector<float64_t> params,
	SGMatrix<float64_t> coeff)
: RandomKitchenSinksDotFeatures(features, D, coeff)
{
	init(kernel_name, params);
}

RandomFourierDotFeatures::RandomFourierDotFeatures(std::shared_ptr<File> loader)
{
	not_implemented(SOURCE_LOCATION);;
}

RandomFourierDotFeatures::RandomFourierDotFeatures(const RandomFourierDotFeatures& orig)
: RandomKitchenSinksDotFeatures(orig)
{
	init(orig.kernel, orig.kernel_params);
}

RandomFourierDotFeatures::~RandomFourierDotFeatures()
{
}

	void RandomFourierDotFeatures::init(
	    KernelName kernel_name, SGVector<float64_t> params)
	{
		kernel = kernel_name;
		kernel_params = params;

		constant = num_samples > 0 ? std::sqrt(2.0 / num_samples) : 1;
		SG_ADD(
		    &kernel_params, "kernel_params",
		    "The parameters of the kernel to approximate");
		SG_ADD(&constant, "constant", "A constant needed");
		SG_ADD_OPTIONS(
		    (machine_int_t*)&kernel, "kernel", "The kernel to approximate",
		    ParameterProperties::NONE, SG_OPTIONS(GAUSSIAN, NOT_SPECIFIED));
	}

std::shared_ptr<Features> RandomFourierDotFeatures::duplicate() const
{
	return std::make_shared<RandomFourierDotFeatures>(*this);
}

const char* RandomFourierDotFeatures::get_name() const
{
	return "RandomFourierDotFeatures";
}

float64_t RandomFourierDotFeatures::post_dot(float64_t dot_result, index_t par_idx) const
{
	dot_result += random_coeff(random_coeff.num_rows-1, par_idx);
	return std::cos(dot_result) * constant;
}

SGVector<float64_t> RandomFourierDotFeatures::generate_random_parameter_vector()
{
	NormalDistribution<float64_t> normal_dist;
	UniformRealDistribution<float64_t> uniform_real_dist(0.0, 2 * Math::PI);
	SGVector<float64_t> vec(feats->get_dim_feature_space()+1);
	switch (kernel)
	{
		case GAUSSIAN:
			for (index_t i=0; i<vec.vlen-1; i++)
			{
				vec[i] = std::sqrt((float64_t)1 / kernel_params[0]) *
				         std::sqrt(2.0) * normal_dist(m_prng);
			}

			vec[vec.vlen-1] = uniform_real_dist(m_prng);
			break;

		default:
			error("Unknown kernel");
	}
	return vec;
}

}
