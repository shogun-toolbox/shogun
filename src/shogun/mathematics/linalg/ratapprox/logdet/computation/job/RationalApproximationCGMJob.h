/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef RATIONAL_APPROXIMATION_CGM_JOB_H_
#define RATIONAL_APPROXIMATION_CGM_JOB_H_

#include <lib/config.h>

#ifdef HAVE_EIGEN3
#include <lib/computation/job/IndependentJob.h>

namespace shogun
{
template<class T> class SGVector;
template<class T> class CLinearOperator;
template<class T> class CStoreScalarAggregator;
class CCGMShiftedFamilySolver;

/** @brief Implementation of independent jobs that solves one whole family
 * of shifted systems in rational approximation of linear operator function
 * times a vector using CG-M linear solver. compute calls submit_results of
 * the aggregator with CScalarResult (see CRationalApproximation)
 */
class CRationalApproximationCGMJob : public CIndependentJob
{
public:
	/** default constructor */
	CRationalApproximationCGMJob();

	/**
	 * constructor
	 *
	 * @param aggregator the scalar job result aggregator for this job
	 * @param linear_solver solver for the shifted-system of this job
	 * @param linear_operator the linear operator of the system
	 * @param vector the sample vector of the system
	 * @param shifts the complex shifts vector in the system
	 * @param weights the complex weights vector in the system
	 * @param const_multiplier the constant multiplier
	 */
	CRationalApproximationCGMJob(CStoreScalarAggregator<float64_t>* aggregator,
		CCGMShiftedFamilySolver* linear_solver,
		CLinearOperator<float64_t>* linear_operator,
		SGVector<float64_t> vector, SGVector<complex128_t> shifts,
		SGVector<complex128_t> weights, float64_t const_multiplier);

	/** destructor */
	virtual ~CRationalApproximationCGMJob();

	/** implementation of compute method for the job */
	virtual void compute();

	/** @return object name */
	virtual const char* get_name() const
	{
		return "RationalApproximationCGMJob";
	}

private:
	/** the real valued linear operator of linear system to be solved */
	CLinearOperator<float64_t>* m_operator;

	/** the vector of the system to be solved */
	SGVector<float64_t> m_vector;

	/** the complex-shifted linear family solver */
	CCGMShiftedFamilySolver* m_linear_solver;

	/** the shifts in the systems to be solved */
	SGVector<complex128_t> m_shifts;

	/** the weights to be multiplied with each solution per shift */
	SGVector<complex128_t> m_weights;

	/** the constant multiplier */
	float64_t m_const_multiplier;

	/** initialize with default values and register params */
	void init();
};

}

#endif // HAVE_EIGEN3
#endif // RATIONAL_APPROXIMATION_CGM_JOB_H_
