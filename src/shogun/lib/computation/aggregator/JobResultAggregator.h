/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef JOB_RESULT_AGGREGATOR_H_
#define JOB_RESULT_AGGREGATOR_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/computation/jobresult/JobResult.h>
#include <shogun/base/Parameter.h>

namespace shogun
{

/** @brief Abstract base class that provides an interface for computing an
 * aggeregation of the job results of independent computation jobs as
 * they are submitted and also for finalizing the aggregation.
 */
class CJobResultAggregator : public CSGObject
{
public:
	/** default constructor */
	CJobResultAggregator()
	: CSGObject()
	{
		init();

		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** destructor */
	virtual ~CJobResultAggregator()
	{
		SG_UNREF(m_result);

		SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
	}

	/**
	 * abstract method that submits the result of an independent job, and
	 * computes the aggregation with the previously submitted result
	 *
	 * @param result the result of an independent job
	 */
	virtual void submit_result(CJobResult* result) = 0;

	/**
	 * abstract method that finalizes the aggregation and computes the result,
	 * its necessary to call finalize before getting the final result
	 */
	virtual void finalize() = 0;

	/** @return the final result */
	CJobResult* get_final_result() const
	{
		return m_result;
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "JobResultAggregator";
	}
protected:
	/** the final job result */
	CJobResult* m_result;

private:
	/** initialize with default values and register params */
	void init()
	{
		m_result=NULL;

		SG_ADD((CSGObject**)&m_result, "final_result",
			"Aggregation of computation job results", MS_NOT_AVAILABLE);
	}
};

}

#endif // JOB_RESULT_AGGREGATOR_H_
