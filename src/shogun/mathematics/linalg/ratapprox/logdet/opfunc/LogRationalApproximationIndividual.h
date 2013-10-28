/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef LOG_RATIONAL_APPROXIMATION_INDIVIDUAL_H_
#define LOG_RATIONAL_APPROXIMATION_INDIVIDUAL_H_

#include <shogun/lib/config.h>
#include <shogun/mathematics/linalg/ratapprox/opfunc/RationalApproximation.h>

#ifdef HAVE_EIGEN3

namespace shogun
{

template<class T> class SGVector;
template<class T> class CMatrixOperator;
template<class T, class ST> class CLinearSolver;
class CJobResultAggregator;
class CIndependentComputationEngine;

/** @brief Implementaion of rational approximation of a operator-function times
 * vector where the operator function is log of a dense-matrix. Each complex
 * system generated from the shifts due to rational approximation of opertor-
 * log times vector expression are solved individually with a complex linear
 * solver by the computation engine. generate_jobs generates num_shifts number of
 * jobs per trace sample
 */
class CLogRationalApproximationIndividual : public CRationalApproximation
{
public:
	/** default constructor */
	CLogRationalApproximationIndividual();

	/**
	 * Constructor. Number of shifts will be computed using a specified accuracy.
	 *
	 * @param linear_operator matrix linear operator of the log function
	 * @param computation_engine engine that computes the independent jobs
	 * @param eigen_solver eigen solver for computing min and max eigenvalues
	 * needed for computing shifts, weights and multiplier in the rational
	 * approximation
	 * @param linear_solver linear solver for solving complex systems
	 * @param desired_accuracy desired error bound on approximation. Computes
	 * the number of shifts automatically
	 */
	CLogRationalApproximationIndividual(
		CMatrixOperator<float64_t>* linear_operator,
		CIndependentComputationEngine* computation_engine,
		CEigenSolver* eigen_solver,
		CLinearSolver<complex128_t, float64_t>* linear_solver,
		float64_t desired_accuracy);

	/** destructor */
	virtual ~CLogRationalApproximationIndividual();

	/**
	 * method that creates a vector job result aggregator, then creates
	 * num_shifts jobs per sample, with each of the shifts moved inside the
	 * linear operator, attaches the aggregator with them, and submits
	 * the job to computation engine and then returns the aggregator
	 *
	 * @param sample the vector for which new computation jobs are to be created
	 * @return the job result aggregator of all the jobs created
	 */
	virtual CJobResultAggregator* submit_jobs(SGVector<float64_t> sample);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "LogRationalApproximationIndividual";
	}

private:
	/** the linear solver for solving complex systems */
	CLinearSolver<complex128_t, float64_t>* m_linear_solver;

	/** initialize with default values and register params */
	void init();
};

}

#endif // HAVE_EIGEN3
#endif // LOG_RATIONAL_APPROXIMATION_INDIVIDUAL_H_
