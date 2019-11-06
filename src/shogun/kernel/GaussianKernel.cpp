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
	require(kernel->get_kernel_type()==K_GAUSSIAN,
		"Provided kernel ({}) must be of type CGaussianKernel!", kernel->get_name());

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
	return CShiftInvariantKernel::init(l, r);
}

void CGaussianKernel::set_width(float64_t w)
{
	require(w>0, "width ({}) must be positive",w);
	m_log_width = std::log(w / 2.0) / 2.0;
	m_eigen_log_width.value() = m_log_width;
}

auto CGaussianKernel::kernel_function(int32_t idx_a, int32_t idx_b)
{
	// this could be written as Eigen::Matrix<float64_t, n_differentiable_params, 1>;
	m_eigen_log_width.value() = m_log_width;

	// this could be written as 
	// eigen_log_width.derivatives() = EigenScalar::Unit(n_differentiable_params, i);
	// where i is the idx of the adjoint
	m_eigen_log_width.derivatives() = Eigen::VectorXd::Unit(1,0);

	auto el = CShiftInvariantKernel::distance(idx_a, idx_b);
	return exp(-el / (exp(m_eigen_log_width * 2.0) * 2.0));
}

SGMatrix<float64_t> CGaussianKernel::get_parameter_gradient(const TParameter* param, index_t index)
{
	using std::exp;

	require(lhs, "Left hand side features must be set!");
	require(rhs, "Right hand side features must be set!");

	if (!strcmp(param->m_name, "log_width"))
	{
		SGMatrix<float64_t> derivative=SGMatrix<float64_t>(num_lhs, num_rhs);
		
		for (int k=0; k<num_rhs; k++)
		{
// #pragma omp parallel for
			for (int j=0; j<num_lhs; j++)
			{
				auto kernel = kernel_function(j, k);
				// 0 is the index of the width parameter
				derivative(j, k) = kernel.derivatives()(0);
			}
		}
		return derivative;
	}
	else
	{
		error("Can't compute derivative wrt {} parameter", param->m_name);
		return SGMatrix<float64_t>();
	}
}

float64_t CGaussianKernel::compute(int32_t idx_a, int32_t idx_b)
{
	auto kernel = kernel_function(idx_a, idx_b);
	return kernel.value();
}	

void CGaussianKernel::load_serializable_post() noexcept(false)
{
	CKernel::load_serializable_post();
	if (lhs && rhs)
		m_distance->init(lhs, rhs);
}

float64_t CGaussianKernel::distance(int32_t idx_a, int32_t idx_b) const
{
	return CShiftInvariantKernel::distance(idx_a, idx_b)/get_width();
}

void CGaussianKernel::register_params()
{
	set_width(1.0);
	set_cache_size(10);

	CEuclideanDistance* dist=new CEuclideanDistance();
	dist->set_disable_sqrt(true);
	m_distance=dist;
	SG_REF(m_distance);

	SG_ADD(&m_log_width, "log_width", "Kernel width in log domain", ParameterProperties::HYPER | ParameterProperties::GRADIENT);
}
