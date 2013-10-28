/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef DENSE_MATRIX_EXACT_LOG_H_
#define DENSE_MATRIX_EXACT_LOG_H_

#include <shogun/lib/config.h>
#include <shogun/mathematics/linalg/ratapprox/opfunc/OperatorFunction.h>

#ifdef HAVE_EIGEN3

namespace shogun
{

template<class T> class SGVector;
template<class T> class CDenseMatrixOperator;
class CJobResultAggregator;
class CIndependentComputationEngine;

/** @brief Class that generates jobs for computing logarithm of
 *  a dense matrix linear operator
 */
class CDenseMatrixExactLog : public COperatorFunction<float64_t>
{
public:
	/** default constructor */
	CDenseMatrixExactLog();

	/**
	 * constructor
	 *
	 * @param op the dense matrix linear operator for this operator function
	 * @param engine the computation engine for the independent jobs
	 */
	CDenseMatrixExactLog(CDenseMatrixOperator<float64_t>* op,
		CIndependentComputationEngine* engine);

	/** destructor */
	virtual ~CDenseMatrixExactLog();

	/**
	 * precompute method that computes the log of the linear operator using
	 * Eigen3, creates a new linear operator and sets that as the operator to
	 * apply to vectors for this operator function
	 */
	virtual void precompute();

	/**
	 * method that creates a scalar job result aggregator, then creates one
	 * job per sample, attaches the aggregator with it, and submits the job to
	 * computation engine and then returns the aggregator
	 *
	 * @param sample the vector for which a new computation job has to be created
	 * @return the array of generated independent jobs
	 */
	virtual CJobResultAggregator* submit_jobs(SGVector<float64_t> sample);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "DenseMatrixExactLog";
	}
};

}

#endif // HAVE_EIGEN3
#endif // DENSE_MATRIX_EXACT_LOG_H_
