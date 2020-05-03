/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/distance/EuclideanDistance.h>
#include <shogun/kernel/MaternKernel.h>

using namespace shogun;

MaternKernel::MaternKernel() : ShiftInvariantKernel()
{
	auto distance = std::make_shared<EuclideanDistance>();
	distance->set_disable_sqrt(false);
	m_distance = distance;
	SG_ADD(
	    &m_nu, "nu", "order of the modified Bessel function of the second kind",
	    ParameterProperties::HYPER)
	SG_ADD(
	    &m_width, "width", "Kernel scale",
	    ParameterProperties::HYPER | ParameterProperties::GRADIENT)
}

MaternKernel::MaternKernel(int32_t size, float64_t w, float64_t order)
    : MaternKernel()
{
	m_nu = order;
	m_width = w;
	set_cache_size(size);
}

MaternKernel::MaternKernel(float64_t w, float64_t order)
    : MaternKernel(10, w, order)
{
}

MaternKernel::MaternKernel(
    const std::shared_ptr<Features>& l, const std::shared_ptr<Features>& r,
    int32_t size, float64_t w, float64_t order)
    : MaternKernel(size, w, order)
{
	init(l, r);
}

MaternKernel::~MaternKernel()
{
	cleanup();
}

void MaternKernel::cleanup()
{
	Kernel::cleanup();
	m_distance->cleanup();
}

bool MaternKernel::init(
    std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	cleanup();
	return ShiftInvariantKernel::init(l, r);
}

float64_t MaternKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result;
	const float64_t dist = ShiftInvariantKernel::distance(idx_a, idx_b);

	// first we can check if we should use one of the approximations which are
	// cheaper to calculate these formulas were taken from Rasmussen's GPs for
	// ML
	if (Math::fequals(m_nu, 0.5, std::numeric_limits<float64_t>::epsilon()))
		result = std::exp(-dist / m_width);
	else if (Math::fequals(
	             m_nu, 1.5, std::numeric_limits<float64_t>::epsilon()))
	{
		float64_t ratio = (std::sqrt(3) * dist) / m_width;
		result = (1 + ratio) * std::exp(-ratio);
	}
	else if (Math::fequals(
	             m_nu, 2.5, std::numeric_limits<float64_t>::epsilon()))
	{
		float64_t ratio = (std::sqrt(5) * dist) / m_width;
		result = (1 + ratio +
		          ((5 * std::pow(dist, 2)) / (3 * std::pow(m_width, 2)))) *
		         std::exp(-ratio);
	}
	// if none of the above was true we compute the value the expensive way
	// in theory if nu is large we could calculate this with the squared
	// exponential covariance function
	else
	{
#ifdef _LIBCPP_VERSION
		error(
		    "Shogun compiled with libc++ does not support Matern kernels with "
		    "width other than 0.5, 1.5 and 2.5. The current width is {}.",
		    m_width);
#else
		const float64_t adjusted_dist =
		    Math::fequals(dist, 0.0, std::numeric_limits<float64_t>::epsilon())
		        ? std::numeric_limits<float32_t>::epsilon()
		        : dist;
		float64_t ratio = std::sqrt(2 * m_nu) * adjusted_dist / m_width;
		// bessel function of the second kind
		float64_t bessel = std::cyl_bessel_k(m_nu, ratio);
		result = (std::pow(2, 1 - m_nu) / std::tgamma(m_nu)) *
		         std::pow(ratio, m_nu) * bessel;
#endif
	}

	return result;
}

SGMatrix<float64_t> MaternKernel::get_parameter_gradient(
    Parameters::const_reference param, index_t index)
{
	require(lhs, "Left hand side features must be set!");
	require(rhs, "Right hand side features must be set!");

	if (param.first == "width")
	{
		SGMatrix<float64_t> derivative = SGMatrix<float64_t>(num_lhs, num_rhs);
		std::function<float64_t(float64_t)> gradient_func;

		// the gradients Matern wrt m_width were computed with WolframAlpha
		if (Math::fequals(m_nu, 0.5, std::numeric_limits<float64_t>::epsilon()))
		{
			const float64_t m_width_squared = std::pow(m_width, 2);
			gradient_func = [&](const float64_t& dist) {
				return (dist * std::exp(-dist / m_width)) / m_width_squared;
			};
		}
		else if (Math::fequals(
		             m_nu, 1.5, std::numeric_limits<float64_t>::epsilon()))
		{
			const float64_t m_width_cubed = std::pow(m_width, 3);
			gradient_func = [&](const float64_t& dist) {
				return (3 * std::pow(dist, 2) *
				        std::exp(-(std::sqrt(3) * dist) / m_width)) /
				       m_width_cubed;
			};
		}
		else if (Math::fequals(
		             m_nu, 2.5, std::numeric_limits<float64_t>::epsilon()))
		{
			const float64_t m_width_power_4 = std::pow(m_width, 4);
			gradient_func = [&](const float64_t& dist) {
				return (5 * std::pow(dist, 2) *
				        std::exp(-(std::sqrt(5) * dist) / m_width) *
				        (m_width + std::sqrt(5) * dist)) /
				       (3 * m_width_power_4);
			};
		}
		else
		{
#ifdef _LIBCPP_VERSION
		error(
		    "Shogun compiled with libc++ does not support Matern kernels with "
		    "width other than 0.5, 1.5 and 2.5. The current width is {}.",
		    m_width);
#else
			gradient_func = [&](const float64_t& dist) {
				constexpr float64_t epsilon = 1E-6;
				auto k = [&](const float64_t& width) {
					const float64_t adjusted_dist =
					    Math::fequals(
					        dist, 0.0,
					        std::numeric_limits<float64_t>::epsilon())
					        ? std::numeric_limits<float32_t>::epsilon()
					        : dist;
					float64_t ratio =
					    std::sqrt(2 * m_nu) * adjusted_dist / width;
					float64_t bessel = std::cyl_bessel_k(m_nu, ratio);
					return (std::pow(2, 1 - m_nu) / std::tgamma(m_nu)) *
					       std::pow(ratio, m_nu) * bessel;
				};
				return (k(m_width + epsilon) - k(m_width)) / epsilon;
			};
#endif
		}

		for (int k = 0; k < num_rhs; k++)
		{
#pragma omp parallel for
			for (int j = 0; j < num_lhs; j++)
			{
				const float64_t dist = ShiftInvariantKernel::distance(j, k);
				derivative(j, k) = gradient_func(dist);
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