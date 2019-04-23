/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Sunil Mahendrakar, Heiko Strathmann, Bjoern Esser
 */

#include <shogun/lib/config.h>

#include <shogun/base/Parameter.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/linop/LinearOperator.h>
#include <shogun/mathematics/linalg/linsolver/CGMShiftedFamilySolver.h>
#include <shogun/mathematics/linalg/ratapprox/logdet/opfunc/LogRationalApproximationCGM.h>

using namespace Eigen;
namespace shogun
{

	LogRationalApproximationCGM::LogRationalApproximationCGM()
	    : RationalApproximation(nullptr, nullptr, 0, OF_LOG)
	{
		init();
}

LogRationalApproximationCGM::LogRationalApproximationCGM(
	std::shared_ptr<LinearOperator<float64_t>> linear_operator, std::shared_ptr<EigenSolver> eigen_solver,
	std::shared_ptr<CGMShiftedFamilySolver> linear_solver, float64_t desired_accuracy)
	: RationalApproximation(
	      linear_operator, eigen_solver, desired_accuracy, OF_LOG)
{
	init();

	m_linear_solver=linear_solver;

}

void LogRationalApproximationCGM::init()
{
	m_linear_solver=NULL;

	SG_ADD((std::shared_ptr<SGObject>*)&m_linear_solver, "linear_solver",
		"Linear solver for complex systems");
}

LogRationalApproximationCGM::~LogRationalApproximationCGM()
{

}

float64_t
LogRationalApproximationCGM::compute(SGVector<float64_t> sample) const
{
	SG_DEBUG("Entering\n");
	REQUIRE(sample.vector, "Sample is not initialized!\n");
	REQUIRE(m_linear_operator, "Operator is not initialized!\n");

	// we need to take the negation of the shifts for this case hence we set
	// negate to true
	SGVector<complex128_t> vec = m_linear_solver->solve_shifted_weighted(
		m_linear_operator, sample, m_shifts, m_weights, true);

	// Take negative (see RationalApproximation for the formula)
	Map<VectorXcd> v(vec.vector, vec.vlen);
	v = -v;

	SGVector<float64_t> agg = m_linear_operator->apply(vec.get_imag());
	float64_t result = linalg::dot(sample, agg);
	result *= m_constant_multiplier;
	return result;
}

}
