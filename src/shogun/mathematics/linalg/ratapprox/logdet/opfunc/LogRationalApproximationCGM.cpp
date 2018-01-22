/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Sunil Mahendrakar, Heiko Strathmann, Björn Esser
 */

#include <shogun/lib/config.h>

#include <shogun/base/Parameter.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/linsolver/CGMShiftedFamilySolver.h>
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationCGM.h>

using namespace Eigen;

namespace shogun
{

	CLogRationalApproximationCGM::CLogRationalApproximationCGM()
	    : CRationalApproximation(NULL, NULL, 0, OF_LOG)
	{
		init();
}

CLogRationalApproximationCGM::CLogRationalApproximationCGM(
	CLinearOperator<float64_t>* linear_operator, CEigenSolver* eigen_solver,
	CCGMShiftedFamilySolver* linear_solver, float64_t desired_accuracy)
	: CRationalApproximation(
	      linear_operator, eigen_solver, desired_accuracy, OF_LOG)
{
	init();

	m_linear_solver=linear_solver;
	SG_REF(m_linear_solver);
}

void CLogRationalApproximationCGM::init()
{
	m_linear_solver=NULL;

	SG_ADD((CSGObject**)&m_linear_solver, "linear_solver",
		"Linear solver for complex systems", MS_NOT_AVAILABLE);

	SG_ADD(&m_negated_shifts, "negated_shifts",
		"Negated shifts", MS_NOT_AVAILABLE);
}

CLogRationalApproximationCGM::~CLogRationalApproximationCGM()
{
	SG_UNREF(m_linear_solver);
}

float64_t CLogRationalApproximationCGM::solve(SGVector<float64_t> sample)
{
	SG_DEBUG("Entering\n");
	REQUIRE(sample.vector, "Sample is not initialized!\n");
	REQUIRE(m_linear_operator, "Operator is not initialized!\n");

	// we need to take the negation of the shifts for this case
	if (m_negated_shifts.vector==NULL)
	{
		m_negated_shifts=SGVector<complex128_t>(m_shifts.vlen);
		Map<VectorXcd> shifts(m_shifts.vector, m_shifts.vlen);
		Map<VectorXcd> negated_shifts(m_negated_shifts.vector, m_negated_shifts.vlen);
		negated_shifts=-shifts;
	}
	REQUIRE(
		m_linear_operator->get_dimension() == sample.vlen,
		"Dimension mismatch! %d vs %d\n", m_linear_operator->get_dimension(),
		sample.vlen);
	REQUIRE(
		m_negated_shifts.vlen == m_weights.vlen,
		"Number of shifts and weights are not equal!\n");

	SGVector<complex128_t> vec = m_linear_solver->solve_shifted_weighted(
		m_linear_operator, sample, m_negated_shifts, m_weights);

	Map<VectorXcd> v(vec.vector, vec.vlen);
	v = -v;
	// take out the imaginary part of the result before
	// applying linear operator
	SGVector<float64_t> agg = m_linear_operator->apply(vec.get_imag());

	// perform dot product
	Map<VectorXd> map_agg(agg.vector, agg.vlen);
	Map<VectorXd> map_vector(sample.vector, sample.vlen);
	float64_t result = map_vector.dot(map_agg);

	result *= m_constant_multiplier;
	return result;
}

}
