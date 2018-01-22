/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Heiko Strathmann, Björn Esser
 */

#include <shogun/lib/config.h>

#include <shogun/base/Parameter.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linop/DenseMatrixOperator.h>
#include <shogun/mathematics/linalg/linop/SparseMatrixOperator.h>
#include <shogun/mathematics/linalg/linsolver/LinearSolver.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationIndividual.h>
#include <typeinfo>

using namespace Eigen;

namespace shogun
{

	CLogRationalApproximationIndividual::CLogRationalApproximationIndividual()
	    : CRationalApproximation(NULL, NULL, 0, OF_LOG)
	{
		init();

		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CLogRationalApproximationIndividual::CLogRationalApproximationIndividual(
	CMatrixOperator<float64_t>* linear_operator, CEigenSolver* eigen_solver,
	CLinearSolver<complex128_t, float64_t>* linear_solver,
	float64_t desired_accuracy)
	: CRationalApproximation(
	      linear_operator, eigen_solver, desired_accuracy, OF_LOG)
{
	init();

	m_linear_solver=linear_solver;
	SG_REF(m_linear_solver);

	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

void CLogRationalApproximationIndividual::init()
{
	m_linear_solver=NULL;

	SG_ADD((CSGObject**)&m_linear_solver, "linear_solver",
		"Linear solver for complex systems", MS_NOT_AVAILABLE);
}

CLogRationalApproximationIndividual::~CLogRationalApproximationIndividual()
{
	SG_UNREF(m_linear_solver);

	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

float64_t CLogRationalApproximationIndividual::solve(SGVector<float64_t> sample)
{
	SG_DEBUG("OperatorFunction::solve(): Entering..\n");
	REQUIRE(sample.vector, "Sample is not initialized!\n");
	REQUIRE(m_linear_operator, "Operator is not initialized!\n");

	// this enum will save from repeated typechecking for all jobs
	enum typeID {DENSE=1, SPARSE, UNKNOWN} operator_type=UNKNOWN;

	// create a complex copy of the matrix linear operator
	CMatrixOperator<complex128_t>* complex_op=NULL;
	if (typeid(*m_linear_operator)==typeid(CDenseMatrixOperator<float64_t>))
	{
		operator_type=DENSE;

		CDenseMatrixOperator<float64_t>* op
			=dynamic_cast<CDenseMatrixOperator<float64_t>*>(m_linear_operator);

		REQUIRE(op->get_matrix_operator().matrix, "Matrix is not initialized!\n");

		// create complex dense matrix operator
		complex_op=static_cast<CDenseMatrixOperator<complex128_t>*>(*op);
	}
	else if (typeid(*m_linear_operator)==typeid(CSparseMatrixOperator<float64_t>))
	{
		operator_type=SPARSE;

		CSparseMatrixOperator<float64_t>* op
			=dynamic_cast<CSparseMatrixOperator<float64_t>*>(m_linear_operator);

		REQUIRE(op->get_matrix_operator().sparse_matrix, "Matrix is not initialized!\n");

		// create complex sparse matrix operator
		complex_op=static_cast<CSparseMatrixOperator<complex128_t>*>(*op);
	}
	else
	{
		// something weird happened
		SG_ERROR("OperatorFunction::solve(): Unknown MatrixOperator given!\n");
	}

	// create num_shifts number of jobs for current sample vector
	for (index_t i=0; i<m_num_shifts; ++i)
	{
		// create a deep copy of the operator
		CMatrixOperator<complex128_t>* shifted_op=NULL;

		switch(operator_type)
		{
		case DENSE:
			shifted_op=new CDenseMatrixOperator<complex128_t>
				(*dynamic_cast<CDenseMatrixOperator<complex128_t>*>(complex_op));
			break;
		case SPARSE:
			shifted_op=new CSparseMatrixOperator<complex128_t>
				(*dynamic_cast<CSparseMatrixOperator<complex128_t>*>(complex_op));
			break;
		default:
			break;
		}

		REQUIRE(
			shifted_op, "OperatorFunction::solve():"
			            "MatrixOperator typeinfo was not detected!\n");

		// move the shift inside the operator
		// (see CRationalApproximation)
		SGVector<complex128_t> diag=shifted_op->get_diagonal();
		for (index_t j=0; j<diag.vlen; ++j)
			diag[j]-=m_shifts[i];
		shifted_op->set_diagonal(diag);

		SGVector<complex128_t> vec = m_linear_solver->solve(shifted_op, sample);

		// multiply with the weight using Eigen3 and take negative
		// (see CRationalApproximation for the formula)
		Map<VectorXcd> v(vec.vector, vec.vlen);
		v *= m_weights[i];
		v = -v;

		SGVector<float64_t> agg = m_linear_operator->apply(vec.get_imag());

		// perform dot product
		Map<VectorXd> map_agg(agg.vector, agg.vlen);
		Map<VectorXd> map_vector(sample.vector, sample.vlen);
		float64_t result = map_vector.dot(map_agg);

		result *= m_constant_multiplier;

		return result;
}

}
}
