/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/distance/EuclideanDistance.h>
#include <shogun/kernel/MaternKernel.h>
#include <shogun/mathematics/bessel.h>
#include <shogun/lib/Fequal.h>

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

MaternKernel::MaternKernel(float64_t w, float64_t order)
    : MaternKernel()
{
	m_nu = order;
	m_width = w;
}

MaternKernel::MaternKernel(
    const std::shared_ptr<Features>& l, const std::shared_ptr<Features>& r,
    float64_t w, float64_t order)
    : MaternKernel(w, order)
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

	// first we check if we should use one of the approximations which are
	// cheaper to calculate
	if (fequals(m_nu, 0.5, std::numeric_limits<float64_t>::epsilon()))
		result = std::exp(-dist / m_width);
	else if (fequals(
	             m_nu, 1.5, std::numeric_limits<float64_t>::epsilon()))
	{
		const float64_t ratio = (std::sqrt(3) * dist) / m_width;
		result = (1 + ratio) * std::exp(-ratio);
	}
	else if (fequals(
	             m_nu, 2.5, std::numeric_limits<float64_t>::epsilon()))
	{
		const float64_t ratio = (std::sqrt(5) * dist) / m_width;
		result = (1 + ratio +
		          ((5 * std::pow(dist, 2)) / (3 * std::pow(m_width, 2)))) *
		         std::exp(-ratio);
	}
	// if none of the above is true we compute the value the expensive way
	// using the modified bessel function of the second kind
	else
	{
		const float64_t adjusted_dist =
		    fequals(dist, 0.0, std::numeric_limits<float64_t>::epsilon())
		        ? std::numeric_limits<float32_t>::epsilon()
		        : dist;
		const float64_t ratio = std::sqrt(2 * m_nu) * adjusted_dist / m_width;
		// bessel function of the second kind
		const float64_t bessel = cyl_bessel_k(m_nu, ratio);
		result = (std::pow(2, 1 - m_nu) / std::tgamma(m_nu)) *
		         std::pow(ratio, m_nu) * bessel;
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

		// the gradients of Matern kernel wrt m_width were computed with
		// WolframAlpha
		if (fequals(m_nu, 0.5, std::numeric_limits<float64_t>::epsilon()))
		{
			const float64_t width_squared = std::pow(m_width, 2);
			gradient_func = [width_squared, width=m_width](const float64_t& dist) {
				const auto upper = dist * std::exp(-dist / width);
				const auto lower = width_squared;
				return upper / lower;
			};
		}
		else if (fequals(
		             m_nu, 1.5, std::numeric_limits<float64_t>::epsilon()))
		{
			const float64_t width_cubed = std::pow(m_width, 3);
			gradient_func = [width_cubed, width=m_width](const float64_t& dist) {
				const auto upper = 3 * std::pow(dist, 2) *
				        std::exp(-(std::sqrt(3) * dist) / width);
				const auto lower = width_cubed;
				return  upper / lower;
			};
		}
		else if (fequals(
		             m_nu, 2.5, std::numeric_limits<float64_t>::epsilon()))
		{
			const float64_t width_power_4 = std::pow(m_width, 4);
			gradient_func = [width_power_4, width = m_width](const float64_t& dist) {
				const auto upper = 5 * std::pow(dist, 2) *
				        std::exp(-(std::sqrt(5) * dist) / width) *
				        (width + std::sqrt(5) * dist);
				const auto lower = 3 * width_power_4;
				return  upper / lower;
			};
		}
		else
		{
			gradient_func = [&](const float64_t& dist) {
				if (fequals(
				        dist, 0.0, std::numeric_limits<float64_t>::epsilon()))
					return 0.0;

				const auto bessel_input =
				    (std::sqrt(2) * dist * std::sqrt(m_nu)) / m_width;
				const auto kv_minus1 = cyl_bessel_k(m_nu - 1, bessel_input);
				const auto kv = cyl_bessel_k(m_nu, bessel_input);
				const auto kv_plus1 = cyl_bessel_k(m_nu + 1, bessel_input);

				const auto ratio = (dist * std::sqrt(m_nu) / m_width);

				float64_t part1 = dist * std::pow(2, 1 - (0.5 * m_nu));
				part1 *= std::pow(m_nu, 1.5);
				part1 *= std::pow(ratio, m_nu - 1);
				part1 *= kv;

				float64_t part2 = dist * std::pow(2, 0.5 - (0.5 * m_nu));
				part2 *= std::sqrt(m_nu);
				part2 *= std::pow(ratio, m_nu);
				part2 *= -kv_minus1 - kv_plus1;

				return (-part1 - part2) /
				       (std::pow(m_width, 2) * std::tgamma(m_nu));
			};
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