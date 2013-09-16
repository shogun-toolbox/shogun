/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 */

#ifndef DENSE_EXACT_LOG_JOB_H_
#define DENSE_EXACT_LOG_JOB_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/computation/job/IndependentJob.h>

namespace shogun
{
template<class T> class SGVector;
template<class T> class CDenseMatrixOperator;

/** @brief Class that represents the job of applying the log of
 * a CDenseMatrixOperator on a real vector
 */
class CDenseExactLogJob : public CIndependentJob
{
public:
	/** default constructor */
	CDenseExactLogJob();

	/** 
	 * constructor
	 *
	 * @param aggregator the job result aggregator for this job
	 * @param log_operator the dense matrix operator to be applied to the vector
	 * @param vector the sample vector to which operator is to be applied
	 */
	CDenseExactLogJob(CJobResultAggregator* aggregator,
		CDenseMatrixOperator<float64_t>* log_operator,
		SGVector<float64_t> vector);

	/** destructor */
	virtual ~CDenseExactLogJob();

	/** implementation of compute method for the job */
	virtual void compute();

	/** @return the vector */
	SGVector<float64_t> get_vector() const;

	/** @return the linear operator */
	CDenseMatrixOperator<float64_t>* get_operator() const;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "DenseExactLogJob";
	}

private:
	/** the log of a CDenseMatrixOperator<float64_t> */
	CDenseMatrixOperator<float64_t>* m_log_operator;

	/** the trace-sample */
	SGVector<float64_t> m_vector;

	/** initialize with default values and register params */
	void init();
};

}

#endif // HAVE_EIGEN3
#endif // DENSE_EXACT_LOG_JOB_H_
