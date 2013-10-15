/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef INDEPENDENT_JOB_H_
#define INDEPENDENT_JOB_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/computation/aggregator/JobResultAggregator.h>

namespace shogun
{

/** @brief Abstract base for general computation jobs to be registered
 * in CIndependentComputationEngine. compute method produces a job result
 * and submits it to the internal JobResultAggregator. Each set of jobs
 * that form a result will share the same job result aggregator.
 */
class CIndependentJob : public CSGObject
{
public:
	/** default constructor*/
	CIndependentJob()
	: CSGObject()
	{
		init();
	}

	/**
	 * constructor
	 *
	 * @param aggregator the job result aggregator for the current job
	 */
	CIndependentJob(CJobResultAggregator* aggregator)
	: CSGObject(), m_aggregator(aggregator)
	{
		init();

		m_aggregator=aggregator;
		SG_REF(m_aggregator);
	}

	/** destructor */
	virtual ~CIndependentJob()
	{
		SG_UNREF(m_aggregator);
	}

	/**
	 * abstract compute method that computes the job, creates a CJobResult,
	 * submits the result to the job result aggregator
	 */
	virtual void compute() = 0;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "IndependentJob";
	}
protected:
	/** the job result aggregator for the current job */
	CJobResultAggregator* m_aggregator;

private:
	/** initialize with default values and register params */
	void init()
	{
		m_aggregator=NULL;

		SG_ADD((CSGObject**)&m_aggregator, "job_result_aggregator",
			"Job result aggregator for current job", MS_NOT_AVAILABLE);
	}
};

}

#endif // INDEPENDENT_JOB_H_
