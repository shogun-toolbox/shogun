/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef LOG_DET_ESTIMATOR_H_
#define LOG_DET_ESTIMATOR_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>

namespace shogun
{
class CTraceSampler;
template<class T> class COperatorFunction;
class CIndependentComputationEngine;
template<class T> class SGVector;
template<class T> class SGMatrix;

/** @brief Class to create unbiased estimators of \f$log(\left|C\right|)=
 * trace(log(C))\f$. For each estimate, it samples trace vectors (one by one)
 * and calls submit_jobs of COperatorFunction, stores the resulting job result
 * aggregator instances, calls wait_for_all of CIndependentComputationEngine
 * to ensure that the job result aggregators are all up to date. Then simply
 * computes running averages over the estimates
 */
class CLogDetEstimator : public CSGObject
{
public:
	/** default constructor */
	CLogDetEstimator();

	/**
	 * constructor for SGMatrix<float64_t>
	 * with serialcomputation engine, CNormalsampler,
	 * CDenseMatrixExactLog as default
	 * @param dense_mat the input Dense matrix
	 */
	 CLogDetEstimator(SGMatrix<float64_t> dense_mat);

	/**
	 * constructor for SGSparseMatrix<float64_t>
	 * with serialcomputation engine,CLogRationalApproximationIndividual as default
	 * CProbingsampler is used when when HAVE_COLPACK is defined 
	 * else CNormalsampler is used,
	 * @param sparse_mat the input Sparse matrix
	 */
	CLogDetEstimator(SGSparseMatrix<float64_t> sparse_mat);

	/**
	 * constructor
	 *
	 * @param trace_sampler the trace sampler
	 * @param operator_log the operator function
	 * @param computation_engine the independent computation engine
	 */
	CLogDetEstimator(CTraceSampler* trace_sampler,
		COperatorFunction<float64_t>* operator_log,
		CIndependentComputationEngine* computation_engine);

	/** destructor */
	virtual ~CLogDetEstimator();

	/**
	 * method that gives num_estimates number of log-det estimates with running
	 * averaging of the estimates
	 *
	 * @param num_estimates the number of log-det estimates to be computed
	 * @return the log-det estimates
	 */
	SGVector<float64_t> sample(index_t num_estimates);

	/**
	 * method that gives num_estimates number of log-det estimates without any
	 * averaging of the estimates
	 *
	 * @param num_estimates the number of log-det estimates to be computed
	 * @return the log-det estimates
	 */
	SGMatrix<float64_t> sample_without_averaging(index_t num_estimates);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "LogDetEstimator";
	}
	
	/**
	*method that returns the trace sampler
	*
	*@return the Trace Sampler
	*/
	CTraceSampler* get_trace_sampler(void) const;
	
	/**
	*method that returns the computation engine
	*@return the independent computation engine
	*/
	CIndependentComputationEngine* get_computation_engine(void) const;
	
	/**
	*method that returns the operator function
	*@return the operator function
	*/
	COperatorFunction<float64_t>* get_operator_function(void) const;
	
private:
	/** the trace sampler */
	CTraceSampler* m_trace_sampler;

	/** the linear operator function, which is log in this case */
	COperatorFunction<float64_t>* m_operator_log;

	/** the computation engine for the independent jobs */
	CIndependentComputationEngine* m_computation_engine;

	/** initialize with default values and register params */
	void init();
};

}

#endif // LOG_DET_ESTIMATOR_H_
