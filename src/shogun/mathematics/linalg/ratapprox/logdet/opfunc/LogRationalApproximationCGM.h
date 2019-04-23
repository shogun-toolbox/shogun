/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Sunil Mahendrakar, Heiko Strathmann, Bjoern Esser
 */

#ifndef LOG_RATIONAL_APPROXIMATION_CGM_H_
#define LOG_RATIONAL_APPROXIMATION_CGM_H_

#include <shogun/lib/config.h>
#include <shogun/mathematics/linalg/ratapprox/opfunc/RationalApproximation.h>


namespace shogun
{

template<class T> class SGVector;
template<class T> class LinearOperator;
class CGMShiftedFamilySolver;

/** @brief Implementaion of rational approximation of a operator-function times
 * vector where the operator function is log of a linear operator. Each complex
 * system generated from the shifts due to rational approximation of opertor-
 * log times vector expression are solved at once with a shifted linear-family
 * solver.
 */
class LogRationalApproximationCGM : public RationalApproximation
{
public:
	/** default constructor */
	LogRationalApproximationCGM();

	/**
	 * Constructor. Number of shifts will be computed using a specified accuracy.
	 *
	 * @param linear_operator linear operator of the log operator function
	 * @param eigen_solver eigen solver for computing min and max eigenvalues
	 * needed for computing shifts, weights and multiplier in the rational
	 * approximation
	 * @param linear_solver linear solver for solving shifted system family
	 * @param desired_accuracy desired error bound on approximation. Computes
	 * the number of shifts automatically
	 */
	LogRationalApproximationCGM(
		std::shared_ptr<LinearOperator<float64_t>> linear_operator,
		std::shared_ptr<EigenSolver> eigen_solver,
		std::shared_ptr<CGMShiftedFamilySolver> linear_solver,
		float64_t desired_accuracy);

	/** destructor */
	virtual ~LogRationalApproximationCGM();

	/**
	 * method that solves the result for a sample
	 */
	virtual float64_t compute(SGVector<float64_t> sample) const;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "LogRationalApproximationCGM";
	}

private:
	/** the linear solver for solving complex systems */
	std::shared_ptr<CGMShiftedFamilySolver> m_linear_solver;

	/** initialize with default values and register params */
	void init();
};

}

#endif // LOG_RATIONAL_APPROXIMATION_CGM_H_
