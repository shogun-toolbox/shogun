/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/kernel/MaternKernel.h>

using namespace shogun;

MaternKernel::MaternKernel(): DistanceKernel(), nu(1.5)
{
	init();
}

MaternKernel::MaternKernel(int32_t size, float64_t w, float64_t order, const std::shared_ptr<Distance>& dist): 
	DistanceKernel(size, w, dist), nu(order)
{
	init();
}

MaternKernel::~MaternKernel()
{
}

bool MaternKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(distance)
	DistanceKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void MaternKernel::init()
{
	SG_ADD(&nu, "nu", "order of the Bessel function of the second kind", ParameterProperties::HYPER)
}

float64_t MaternKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result;

	float64_t dist = distance->distance(idx_a, idx_b);

	// first we can check if we should use one of the approximations which are cheaper to calculate
	// these formulas were taken from Rasmussen's GPs for ML
	if (Math::fequals(nu, 0.5, std::numeric_limits<float64_t>::epsilon()))
		result = std::exp(-dist/width);
	else if (Math::fequals(nu, 1.5, std::numeric_limits<float64_t>::epsilon()))
	{
		float64_t ratio = (std::sqrt(3) * dist) / width;
		result = (1 + ratio) * std::exp(-ratio);
	}
	else if (Math::fequals(nu, 2.5, std::numeric_limits<float64_t>::epsilon()))
	{
		float64_t ratio = (std::sqrt(5) * dist) / width;
		result = (1 + ratio + ((5 * std::pow(dist, 2)) / (3 * std::pow(width, 2)))) * std::exp(-ratio);
	}
	// if none of the above was true we compute the value the expensive way
	// in theory if nu is large we could calculate this with the squared exponential covariance function
	else
	{
		float64_t ratio = std::sqrt(2 * nu) * dist / width;
		// bessel function of the second kind
#ifdef __clang__
		// clang hasn't implemented mathematical special functions in the C++17 standard at this point
		// note that yn casts nu to an integer
		float64_t bessel = yn(nu, ratio);
#else
		float64_t bessel = std::cyl_neumann(nu, ratio);
#endif
		result = (std::pow(2, 1 - nu) / std::tgamma(nu)) * std::pow(ratio, nu) * bessel;
	}

	return result;
}


SGMatrix<float64_t> MaternKernel::get_parameter_gradient(Parameters::const_reference param, index_t index)
{
	require(lhs, "Left hand side features must be set!");
	require(rhs, "Right hand side features must be set!");

	if (param.first == "width")
	{
		SGMatrix<float64_t> derivative=SGMatrix<float64_t>(num_lhs, num_rhs);
		std::function<float64_t(float64_t)> gradient_func;
		
		// the gradients Matern wrt width were computed with WolframAlpha
		if (Math::fequals(nu, 0.5, std::numeric_limits<float64_t>::epsilon()))
		{
			const float64_t width_squared = std::pow(width, 2);
			gradient_func = [&](const float64_t& dist){ 
				return (dist * std::exp(-dist / width)) / width_squared;
			};
		}
		else if (Math::fequals(nu, 1.5, std::numeric_limits<float64_t>::epsilon()))
		{
			const float64_t width_cubed = std::pow(width, 3);
			gradient_func = [&](const float64_t& dist){ 
					return (3 * std::pow(dist, 2) * std::exp(-(std::sqrt(3) * dist) / width)) / width_cubed; 
			};
		}
		else if (Math::fequals(nu, 2.5, std::numeric_limits<float64_t>::epsilon()))
		{
			const float64_t width_power_4 = std::pow(width, 4);
			gradient_func = [&](const float64_t& dist){ 
				return (5 * std::pow(dist, 2) * std::exp(-(std::sqrt(5) * dist) / width) * (width + std::sqrt(5) * dist)) 
						/ (3 * width_power_4);
			};
		}
		else
		{
			gradient_func = [&](const float64_t& dist){
				const auto ratio = (std::sqrt(2) * dist * std::sqrt(nu)) / width;
#ifdef __clang__
				auto bessel_nu_minus_1 = yn(nu - 1, ratio);
				auto bessel_nu_plus_1 = yn(nu + 1, ratio);
#else
				auto bessel_nu_minus_1 = std::cyl_neumann(nu - 1, ratio);
				auto bessel_nu_plus_1 = std::cyl_neumann(nu + 1, ratio);
#endif
				auto gradient = std::pow(2, (3 * nu) / 2 - 3 / 2);
				gradient *= std::sqrt(nu) * dist;
				gradient *= std::pow((std::sqrt(nu) * dist) / width, nu);
				gradient *= bessel_nu_minus_1 - bessel_nu_plus_1;
				gradient /= std::pow(width, 2) * std::tgamma(nu);

				return gradient;
			};
		}

		for (int k=0; k<num_rhs; k++)
		{
#pragma omp parallel for
			for (int j=0; j<num_lhs; j++)
			{
				const float64_t element = compute(j, k);
				derivative(j, k) = gradient_func(element);
			}
		}
		return derivative;
	}
	else
	{
		error("Can't compute derivative wrt {} parameter", param.first);
		return SGMatrix<float64_t>();
	}
}