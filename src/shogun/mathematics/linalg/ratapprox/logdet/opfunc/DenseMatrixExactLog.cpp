/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sunil Mahendrakar, Soumyajit De, Heiko Strathmann, Bjoern Esser,
 *          Viktor Gal
 */

#include <shogun/lib/common.h>

#include <shogun/mathematics/eigen3.h>

#if EIGEN_VERSION_AT_LEAST(3,1,0)
#include <unsupported/Eigen/MatrixFunctions>
#endif // EIGEN_VERSION_AT_LEAST(3,1,0)
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/DenseMatrixExactLog.h>

using namespace Eigen;

namespace shogun
{

	DenseMatrixExactLog::DenseMatrixExactLog()
	    : OperatorFunction<float64_t>(nullptr, OF_LOG)
	{
	}

	DenseMatrixExactLog::DenseMatrixExactLog(
	    std::shared_ptr<DenseMatrixOperator<float64_t>> op)
	    : OperatorFunction<float64_t>(op->as<LinearOperator<float64_t>>(), OF_LOG)
	{
	}

	DenseMatrixExactLog::~DenseMatrixExactLog()
	{
	}

#if EIGEN_VERSION_AT_LEAST(3,1,0)
void DenseMatrixExactLog::precompute()
{
	SG_TRACE("Entering...");

	// check for proper downcast
	auto op
		=m_linear_operator->as<DenseMatrixOperator<float64_t>>();
	require(op, "Operator not an instance of DenseMatrixOperator!");
	SGMatrix<float64_t> m=op->get_matrix_operator();

	// compute log(C) using Eigen3
	Map<MatrixXd> mat(m.matrix, m.num_rows, m.num_cols);
	SGMatrix<float64_t> log_m(m.num_rows, m.num_cols);
	Map<MatrixXd> log_mat(log_m.matrix, log_m.num_rows, log_m.num_cols);
#if EIGEN_WITH_LOG_BUG_1229
	MatrixXd tmp = mat;
	log_mat=tmp.log();
#else
	log_mat=mat.log();
#endif

	// the log(C) is also a linear operator here
	// reset the operator of this function with log(C)

	m_linear_operator=std::make_shared<DenseMatrixOperator<float64_t>>(log_m);


	SG_TRACE("Leaving...");
}
#else
void DenseMatrixExactLog::precompute()
{
	io::warn("Eigen3.1.0 or later required!");
}
#endif // EIGEN_VERSION_AT_LEAST(3,1,0)

float64_t DenseMatrixExactLog::compute(SGVector<float64_t> sample) const
{
	SG_TRACE("Entering...");

	auto m_log_operator =
		m_linear_operator->as<DenseMatrixOperator<float64_t>>();

	SGVector<float64_t> vec = m_log_operator->apply(sample);
	float64_t result = linalg::dot(sample, vec);
	SG_TRACE("Leaving...");
	return result;
}

}
