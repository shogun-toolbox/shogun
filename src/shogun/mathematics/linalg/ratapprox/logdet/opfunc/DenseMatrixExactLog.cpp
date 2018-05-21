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

	CDenseMatrixExactLog::CDenseMatrixExactLog()
	    : COperatorFunction<float64_t>(nullptr, OF_LOG)
	{
	}

	CDenseMatrixExactLog::CDenseMatrixExactLog(
	    CDenseMatrixOperator<float64_t>* op)
	    : COperatorFunction<float64_t>((CLinearOperator<float64_t>*)op, OF_LOG)
	{
	}

	CDenseMatrixExactLog::~CDenseMatrixExactLog()
	{
	}

#if EIGEN_VERSION_AT_LEAST(3,1,0)
void CDenseMatrixExactLog::precompute()
{
	SG_DEBUG("Entering...\n");

	// check for proper downcast
	CDenseMatrixOperator<float64_t>* op
		=dynamic_cast<CDenseMatrixOperator<float64_t>*>(m_linear_operator);
	REQUIRE(op, "Operator not an instance of DenseMatrixOperator!\n");
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
	SG_UNREF(m_linear_operator);
	m_linear_operator=new CDenseMatrixOperator<float64_t>(log_m);
	SG_REF(m_linear_operator);

	SG_DEBUG("Leaving...\n");
}
#else
void CDenseMatrixExactLog::precompute()
{
	SG_WARNING("Eigen3.1.0 or later required!\n")
}
#endif // EIGEN_VERSION_AT_LEAST(3,1,0)

float64_t CDenseMatrixExactLog::compute(SGVector<float64_t> sample) const
{
	SG_DEBUG("Entering...\n");

	CDenseMatrixOperator<float64_t>* m_log_operator =
		dynamic_cast<CDenseMatrixOperator<float64_t>*>(m_linear_operator);

	SGVector<float64_t> vec = m_log_operator->apply(sample);
	float64_t result = linalg::dot(sample, vec);
	SG_DEBUG("Leaving...\n");
	return result;
}

}
