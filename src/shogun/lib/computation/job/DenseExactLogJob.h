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
	/** default constructor, no args */
	CDenseExactLogJob();

	/** default constructor, one arg */
	CDenseExactLogJob(CJobResultAggregator* aggregator);

	/** destructor */
	virtual ~CDenseExactLogJob();

	/** implementation of compute method for the job */
	virtual void compute();

	/** get the vector */
	SGVector<float64_t> get_vector() const;

	/** set the vector */
	void set_vector(SGVector<float64_t> vec);

	/** get the linear operator */
	CDenseMatrixOperator<float64_t>* get_operator() const;

	/** set the linear operator */
	void set_operator(CDenseMatrixOperator<float64_t>* op);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "CDenseExactLogJob";
	}

private:
	/** the log of a CDenseMatrixOperator<float64_t> */
	CDenseMatrixOperator<float64_t>* m_log_operator;

	/** the trace-sample */
	SGVector<float64_t> m_vector;
};

}

#endif // HAVE_EIGEN3
#endif // DENSE_EXACT_LOG_JOB_H_
