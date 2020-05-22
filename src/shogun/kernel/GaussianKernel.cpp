/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Jacob Walker, Chiyuan Zhang, Wu Lin,
 *          Heiko Strathmann, Roman Votyakov, Soumyajit De, Viktor Gal,
 *          Tonmoy Saikia, Sergey Lisitsyn, Matt Aasted, Sanuj Sharma
 */

#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/lib/auto_initialiser.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

GaussianKernel::GaussianKernel() : ShiftInvariantKernel()
{
	set_cache_size(10);

	auto dist=std::make_shared<EuclideanDistance>();
	dist->set_disable_sqrt(true);
	m_distance=dist;

	SG_ADD(
	    &m_log_width, "log_width", "Kernel width in log domain",
	    ParameterProperties::AUTO | ParameterProperties::HYPER |
	        ParameterProperties::GRADIENT,
	    std::make_shared<params::GaussianWidthAutoInit>(*this, 1.0));
}

GaussianKernel::GaussianKernel(float64_t w) : GaussianKernel()
{
	set_width(w);
}

GaussianKernel::GaussianKernel(int32_t size, float64_t w) : GaussianKernel(w)
{
	set_cache_size(size);
}

GaussianKernel::GaussianKernel(const std::shared_ptr<DotFeatures>& l, const std::shared_ptr<DotFeatures>& r, float64_t w, int32_t size) : GaussianKernel(size, w)
{
	set_cache_size(size);
	set_width(w);
	init(l, r);
}

GaussianKernel::~GaussianKernel()
{
	cleanup();
}

std::shared_ptr<GaussianKernel> GaussianKernel::obtain_from_generic(const std::shared_ptr<Kernel>& kernel)
{
	require(kernel->get_kernel_type()==K_GAUSSIAN,
		"Provided kernel ({}) must be of type GaussianKernel!", kernel->get_name());

	return std::static_pointer_cast<GaussianKernel>(kernel);
}

#include <typeinfo>
std::shared_ptr<SGObject >GaussianKernel::shallow_copy() const
{
	// TODO: remove this after all the classes get shallow_copy properly implemented
	// this assert is to avoid any subclass of GaussianKernel accidentally called
	// with the implement here
	ASSERT(typeid(*this) == typeid(GaussianKernel))
	std::shared_ptr<GaussianKernel> ker;
	if (std::holds_alternative<AutoValueEmpty>(m_log_width)) 
	{
		ker = std::make_shared<GaussianKernel>();		
		ker->set_cache_size(cache_size);
	}
	else
		ker = std::make_shared<GaussianKernel>(cache_size, get_width());

	if (lhs && rhs)
	{
		ker->init(lhs, rhs);
		ker->m_distance->init(lhs, rhs);
	}
	return ker;
}

void GaussianKernel::cleanup()
{
	Kernel::cleanup();
	m_distance->cleanup();
}

bool GaussianKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	cleanup();
	return ShiftInvariantKernel::init(l, r);
}

void GaussianKernel::set_width(float64_t w)
{
	require(w>0, "width ({}) must be positive",w);
	m_log_width = GaussianKernel::to_log_width(w);
}

SGMatrix<float64_t> GaussianKernel::get_parameter_gradient(Parameters::const_reference param, index_t index)
{
	require(lhs, "Left hand side features must be set!");
	require(rhs, "Rightt hand side features must be set!");

	if (param.first == "log_width")
	{
		SGMatrix<float64_t> derivative=SGMatrix<float64_t>(num_lhs, num_rhs);
		for (int k=0; k<num_rhs; k++)
		{
#pragma omp parallel for
			for (int j=0; j<num_lhs; j++)
			{
				float64_t element=distance(j, k);
				derivative(j, k) = std::exp(-element) * element * 2.0;
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

float64_t GaussianKernel::compute(int32_t idx_a, int32_t idx_b)
{
    float64_t result=distance(idx_a, idx_b);
	return std::exp(-result);
}

void GaussianKernel::load_serializable_post() noexcept(false)
{
	Kernel::load_serializable_post();
	if (lhs && rhs)
		m_distance->init(lhs, rhs);
}

float64_t GaussianKernel::distance(int32_t idx_a, int32_t idx_b) const
{
	return ShiftInvariantKernel::distance(idx_a, idx_b)/get_width();
}
