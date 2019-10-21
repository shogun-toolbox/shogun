/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/kernel/MaternKernel.h>

#include <cmath>

CMaternKernel::CMaternKernel(): CDistanceKernel(), nu(0.0)
{
	init();
}

CMaternKernel::CMaternKernel(int32_t size, float64_t w, float64_t order, CDistance* dist): 
	CDistanceKernel(size, w, dist), nu(order)
{
	ASSERT(distance)
	SG_REF(distance);
	init();
}

CMaternKernel::~CMaternKernel()
{
	cleanup();
	SG_UNREF(distance);
}

void CMaternKernel::cleanup()
{
}

bool CMaternKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance)
	CDistanceKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CMaternKernel::init()
{
	SG_ADD(&nu, "order", "order of the Bessel function of the second kind", ParameterProperties::HYPER)
}

float64_t CMaternKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result;

	float64_t dist = distance->distance(idx_a, idx_b);

	// first we can check if we should use one of the approximations which are cheaper to calculate
	if (CMath::fequals(nu, 0.5, std::numeric_limits<float64_t>::epsilon()))
		result = std::exp(-dist/width);
	else if (CMath::fequals(nu, 1.5, std::numeric_limits<float64_t>::epsilon()))
	{
		float64_t ratio = (std::sqrt(3) * dist) / width;
		result = (1 + ratio) * std::exp(-ratio);
	}
	else if (CMath::fequals(nu, 2.5, std::numeric_limits<float64_t>::epsilon()))
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
		float64_t bessel = std::cyl_neumann(nu, ratio);
		result = std::pow(2, 1 - nu) / std::tgamma(nu) * std::pow(ratio, nu) * bessel;
	}

	return result;
}
