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

CGaussianKernel::CGaussianKernel() : CShiftInvariantKernel()
{
	register_params();
}

CGaussianKernel::CGaussianKernel(float64_t w) : CShiftInvariantKernel()
{
	register_params();
	set_width(w);
}

CGaussianKernel::CGaussianKernel(int32_t size, float64_t w) : CShiftInvariantKernel()
{
	register_params();
	set_width(w);
	set_cache_size(size);
}

CGaussianKernel::CGaussianKernel(CDotFeatures* l, CDotFeatures* r, float64_t w, int32_t size) : CShiftInvariantKernel()
{
	register_params();
	set_width(w);
	set_cache_size(size);
	init(l, r);
}

CGaussianKernel::~CGaussianKernel()
{
	cleanup();
}

CGaussianKernel* CGaussianKernel::obtain_from_generic(CKernel* kernel)
{
	REQUIRE(kernel->get_kernel_type()==K_GAUSSIAN,
		"Provided kernel (%s) must be of type CGaussianKernel!\n", kernel->get_name());

	SG_REF(kernel);
	return (CGaussianKernel*)kernel;
}

#include <typeinfo>
CSGObject *CGaussianKernel::shallow_copy() const
{
	// TODO: remove this after all the classes get shallow_copy properly implemented
	// this assert is to avoid any subclass of CGaussianKernel accidentally called
	// with the implement here
	ASSERT(typeid(*this) == typeid(CGaussianKernel))
	CGaussianKernel *ker = new CGaussianKernel(cache_size, get_width());
	if (lhs && rhs)
	{
		ker->init(lhs, rhs);
		ker->m_distance->init(lhs, rhs);
	}
	return ker;
}

void CGaussianKernel::cleanup()
{
	CKernel::cleanup();
	m_distance->cleanup();
}

bool CGaussianKernel::init(CFeatures* l, CFeatures* r)
{
	cleanup();
	CShiftInvariantKernel::init(l, r);
	return init_normalizer();
}

void CGaussianKernel::set_width(float64_t w)
{
	REQUIRE(w>0, "width (%f) must be positive\n",w);
	m_log_width = std::log(w / 2.0) / 2.0;
}

float64_t CGaussianKernel::get_width() const
{
	return CMath::exp(m_log_width*2.0)*2.0;
}

SGMatrix<float64_t> CGaussianKernel::get_parameter_gradient(const TParameter* param, index_t index)
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
				derivative(j, k)=CMath::exp(-element)*element*2.0;
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

float64_t CGaussianKernel::compute(int32_t idx_a, int32_t idx_b)
{
    float64_t result=distance(idx_a, idx_b);
    return CMath::exp(-result);
}

void CGaussianKernel::load_serializable_post() throw (ShogunException)
{
	CKernel::load_serializable_post();
	if (lhs && rhs)
		m_distance->init(lhs, rhs);
}

float64_t CGaussianKernel::distance(int32_t idx_a, int32_t idx_b) const
{
	const float64_t inv_width=1.0/get_width();
	return CShiftInvariantKernel::distance(idx_a, idx_b)*inv_width;
}

void CGaussianKernel::register_params()
{
	set_width(1.0);
	set_cache_size(10);

	CEuclideanDistance* dist=new CEuclideanDistance();
	dist->set_disable_sqrt(true);
	m_distance=dist;
	SG_REF(m_distance);

	SG_ADD(&m_log_width, "log_width", "Kernel width in log domain", MS_AVAILABLE, GRADIENT_AVAILABLE);
}
