/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Heiko Strathmann, Bjoern Esser
 */

#ifndef LOG_RATIONAL_APPROXIMATION_INDIVIDUAL_H_
#define LOG_RATIONAL_APPROXIMATION_INDIVIDUAL_H_

#include <shogun/lib/config.h>
#include <shogun/mathematics/linalg/ratapprox/opfunc/RationalApproximation.h>


namespace shogun
{

template<class T> class SGVector;
template<class T> class MatrixOperator;
template<class T, class ST> class LinearSolver;

/** @brief Implementaion of rational approximation of a operator-function times
 * vector where the operator function is log of a dense-matrix. Each complex
 * system generated from the shifts due to rational approximation of opertor-
 * log times vector expression are solved individually with a complex linear
 * solver.
 */
class LogRationalApproximationIndividual : public RationalApproximation
{
public:
	/** default constructor */
	LogRationalApproximationIndividual();

	/**
	 * Constructor. Number of shifts will be computed using a specified accuracy.
	 *
	 * @param linear_operator matrix linear operator of the log function
	 * @param eigen_solver eigen solver for computing min and max eigenvalues
	 * needed for computing shifts, weights and multiplier in the rational
	 * approximation
	 * @param linear_solver linear solver for solving complex systems
	 * @param desired_accuracy desired error bound on approximation. Computes
	 * the number of shifts automatically
	 */
	LogRationalApproximationIndividual(
		const std::shared_ptr<MatrixOperator<float64_t>>& linear_operator,
		std::shared_ptr<EigenSolver> eigen_solver,
		std::shared_ptr<LinearSolver<complex128_t, float64_t>> linear_solver,
		float64_t desired_accuracy);

	/** destructor */
	~LogRationalApproximationIndividual() override;

	/**
	 *method that solves the result for a sample
	 */
	float64_t compute(SGVector<float64_t> sample) const override;

	/** @return object name */
	const char* get_name() const override
	{
		return "LogRationalApproximationIndividual";
	}

private:
	/** the linear solver for solving complex systems */
	std::shared_ptr<LinearSolver<complex128_t, float64_t>> m_linear_solver;

	/** initialize with default values and register params */
	void init();
};

}

#endif // LOG_RATIONAL_APPROXIMATION_INDIVIDUAL_H_
