/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Jacob Walker, Chiyuan Zhang, Wu Lin,
 *          Heiko Strathmann, Roman Votyakov, Soumyajit De, Viktor Gal,
 *          Tonmoy Saikia, Sergey Lisitsyn, Matt Aasted, Sanuj Sharma
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

GaussianKernel::GaussianKernel() : ShiftInvariantKernel()
{
	register_params();
}

GaussianKernel::GaussianKernel(float64_t w) : ShiftInvariantKernel()
{
	register_params();
	set_width(w);
}

GaussianKernel::GaussianKernel(int32_t size, float64_t w) : ShiftInvariantKernel()
{
	register_params();
	set_width(w);
	set_cache_size(size);
}

GaussianKernel::GaussianKernel(std::shared_ptr<DotFeatures> l, std::shared_ptr<DotFeatures> r, float64_t w, int32_t size) : ShiftInvariantKernel()
{
	register_params();
	set_width(w);
	set_cache_size(size);
	init(l, r);
}

GaussianKernel::~GaussianKernel()
{
	cleanup();
}

std::shared_ptr<GaussianKernel> GaussianKernel::obtain_from_generic(std::shared_ptr<Kernel> kernel)
{
	REQUIRE(kernel->get_kernel_type()==K_GAUSSIAN,
		"Provided kernel (%s) must be of type GaussianKernel!\n", kernel->get_name());


	return std::static_pointer_cast<GaussianKernel>(kernel);
}

#include <typeinfo>
std::shared_ptr<SGObject >GaussianKernel::shallow_copy() const
{
	// TODO: remove this after all the classes get shallow_copy properly implemented
	// this assert is to avoid any subclass of GaussianKernel accidentally called
	// with the implement here
	ASSERT(typeid(*this) == typeid(GaussianKernel))
	auto ker = std::make_shared<GaussianKernel>(cache_size, get_width());
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
	REQUIRE(w>0, "width (%f) must be positive\n",w);
	m_log_width = std::log(w / 2.0) / 2.0;
}

SGMatrix<float64_t> GaussianKernel::get_parameter_gradient(const TParameter* param, index_t index)
{
	REQUIRE(lhs, "Left hand side features must be set!\n")
	REQUIRE(rhs, "Rightt hand side features must be set!\n")

	if (!strcmp(param->m_name, "log_width"))
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
		SG_ERROR("Can't compute derivative wrt %s parameter\n", param->m_name);
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

void GaussianKernel::register_params()
{
	set_width(1.0);
	set_cache_size(10);

	auto dist=std::make_shared<EuclideanDistance>();
	dist->set_disable_sqrt(true);
	m_distance=dist;


	SG_ADD(&m_log_width, "log_width", "Kernel width in log domain", ParameterProperties::HYPER | ParameterProperties::GRADIENT);
}
