/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef RATIONAL_APPROXIMATION_INDIVIDUAL_JOB_H_
#define RATIONAL_APPROXIMATION_INDIVIDUAL_JOB_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/computation/job/IndependentJob.h>

namespace shogun
{
template<class T> class SGVector;
template<class T> class CLinearOperator;
template<class T, class ST> class CLinearSolver;

/** @brief Implementation of independent job that solves one of the family
 * of shifted systems in rational approximation of linear operator function
 * times a vector using a direct linear solver. The shift is moved inside the
 * operator. compute calls submit_results of the aggregator with CVectorResult
 * which is the solution vector for that shift multiplied by complex weight
 * (See CRationalApproximation)
 */
class CRationalApproximationIndividualJob : public CIndependentJob
{
public:
	/** default constructor */
	CRationalApproximationIndividualJob();

	/** 
	 * constructor
	 *
	 * @param aggregator the job result aggregator for this job
	 * @param linear_solver solver for the complex system of this job
	 * @param linear_operator the shifted operator of the system
	 * @param vector the sample vector of the system
	 * @param weight the complex weight to be multiplied with the solution
	 * vector after solving the linear-system
	 */
	CRationalApproximationIndividualJob(CJobResultAggregator* aggregator,
		CLinearSolver<complex128_t, float64_t>* linear_solver,
		CLinearOperator<complex128_t>* linear_operator,
		SGVector<float64_t> vector, complex128_t weight);

	/** destructor */
	virtual ~CRationalApproximationIndividualJob();

	/** implementation of compute method for the job */
	virtual void compute();

	/** @return object name */
	virtual const char* get_name() const
	{
		return "RationalApproximationIndividualJob";
	}

private:
	/** the shifted operator for linear system to be solved */
	CLinearOperator<complex128_t>* m_operator;

	/** the vector of the system to be solved */
	SGVector<float64_t> m_vector;

	/** the complex linear solver for solving systems */
	CLinearSolver<complex128_t, float64_t>* m_linear_solver;

	/** the weight to be multiplied with the solution vector */
	complex128_t m_weight;

	/** initialize with default values and register params */
	void init();
};

}

#endif // HAVE_EIGEN3
#endif // RATIONAL_APPROXIMATION_INDIVIDUAL_JOB_H_
