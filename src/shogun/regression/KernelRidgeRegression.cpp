/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Michal Uricar, Abhinav Rai, 
 *          Roman Votyakov, Sanuj Sharma
 */

#include <shogun/lib/config.h>

#include <shogun/regression/KernelRidgeRegression.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

KernelRidgeRegression::KernelRidgeRegression()
: KernelMachine()
{
	init();
}

KernelRidgeRegression::KernelRidgeRegression(float64_t tau, std::shared_ptr<Kernel> k, std::shared_ptr<Labels> lab)
: KernelMachine()
{
	init();

	set_tau(tau);
	set_labels(lab);
	set_kernel(k);
}

void KernelRidgeRegression::init()
{
	set_tau(1e-6);
	set_epsilon(0.0001);
	SG_ADD(&m_tau, "tau", "Regularization parameter", ParameterProperties::HYPER);
}

bool KernelRidgeRegression::solve_krr_system()
{
	SGMatrix<float64_t> kernel_matrix(kernel->get_kernel_matrix());
	int32_t n = kernel_matrix.num_rows;
	SGVector<float64_t> y = regression_labels(m_labels)->get_labels();

	for(index_t i=0; i<n; i++)
		kernel_matrix(i,i) += m_tau;

	Map<MatrixXd> eigen_kernel_matrix(kernel_matrix.matrix, n, n);
	Map<VectorXd> eigen_alphas(m_alpha.vector, n);
	Map<VectorXd> eigen_y(y.vector, n);

	LLT<MatrixXd> llt;
	llt.compute(eigen_kernel_matrix);
	if (llt.info() != Eigen::Success)
	{
		SG_WARNING("Features covariance matrix was not positive definite\n");
		return false;
	}
	eigen_alphas = llt.solve(eigen_y);
	return true;
}

bool KernelRidgeRegression::train_machine(std::shared_ptr<Features >data)
{
	REQUIRE(m_labels, "No labels set\n")

	if (data)
	{
		if (m_labels->get_num_labels() != data->get_num_vectors())
			SG_ERROR("Number of training vectors does not match number of labels\n")
		kernel->init(data, data);
	}
	ASSERT(kernel && kernel->has_features())

	if (m_labels->get_num_labels() != kernel->get_num_vec_rhs())
	{
		SG_ERROR("Number of labels does not match number of kernel"
			" columns (num_labels=%d cols=%d\n", m_labels->get_num_labels(), kernel->get_num_vec_rhs());
	}

	// allocate alpha vector
	set_alphas(SGVector<float64_t>(m_labels->get_num_labels()));

	if(!solve_krr_system())
		return false;

	/* tell kernel machine that all alphas are needed as'support vectors' */
	m_svs = SGVector<index_t>(m_alpha.vlen);
	m_svs.range_fill();
	return true;
}

bool KernelRidgeRegression::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool KernelRidgeRegression::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}
