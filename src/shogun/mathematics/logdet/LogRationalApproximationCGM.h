/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef LOG_RATIONAL_APPROXIMATION_CGM_H_
#define LOG_RATIONAL_APPROXIMATION_CGM_H_

#include <shogun/lib/config.h>
#include <shogun/mathematics/logdet/RationalApproximation.h>

#ifdef HAVE_EIGEN3

namespace shogun
{

template<class T> class SGVector;
template<class T> class CLinearOperator;
class CCGMShiftedFamilySolver;
class CJobResultAggregator;
class CIndependentComputationEngine;

/** @brief Implementaion of rational approximation of a operator-function times
 * vector where the operator function is log of a linear operator. Each complex
 * system generated from the shifts due to rational approximation of opertor-
 * log times vector expression are solved at once with a shifted linear-family
 * solver by the computation engine. generate_jobs generates one job per sample
 */
class CLogRationalApproximationCGM : public CRationalApproximation
{
public:
	/** default constructor */
	CLogRationalApproximationCGM();

	/** 
	 * constructor
	 *
	 * @param linear_operator linear operator of the log operator function
	 * @param computation_engine engine that computes the independent jobs
	 * @param eigen_solver eigen solver for computing min and max eigenvalues
	 * needed for computing shifts, weights and multiplier in the rational
	 * approximation
	 * @param linear_solver linear solver for solving shifted system family
	 * @param num_shifts number of contour points in the quadrature rule of
	 * of discretization of the contour integral
	 */
	CLogRationalApproximationCGM(
		CLinearOperator<float64_t>* linear_operator,
		CIndependentComputationEngine* computation_engine,
		CEigenSolver* eigen_solver,
		CCGMShiftedFamilySolver* linear_solver,
		index_t num_shifts);

	/** destructor */
	virtual ~CLogRationalApproximationCGM();

	/** 
	 * method that creates a scalar job result aggregator, then creates 
	 * one job per trace sample, attaches the aggregator with them, and submits
	 * the job to computation engine and then returns the aggregator
	 *
	 * @param sample the vector for which new computation jobs are to be created
	 * @return the job result aggregator of all the jobs created
	 */
	virtual CJobResultAggregator* submit_jobs(SGVector<float64_t> sample);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "LogRationalApproximationCGM";
	}

private:
	/** the linear solver for solving complex systems */
	CCGMShiftedFamilySolver* m_linear_solver;

	/** negated shifts to pass to CG-M linear solver */
	SGVector<complex64_t> m_negated_shifts;

	/** initialize with default values and register params */
	void init();
};

}

#endif // HAVE_EIGEN3
#endif // LOG_RATIONAL_APPROXIMATION_CGM_H_
