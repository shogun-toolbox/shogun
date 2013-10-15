/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef INDEPENDENT_COMPUTATION_ENGINE_H_
#define INDEPENDENT_COMPUTATION_ENGINE_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>

namespace shogun
{
class CIndependentJob;

/** @brief Abstract base class for solving multiple independent instances of
 * CIndependentJob. It has one method, submit_job, which may add the job to an
 * internal queue and might block if there is yet not space in the queue.
 * After jobs are submitted, it might not yet be ready. wait_for_all waits
 * until all jobs are completed, which *must be* called to guarantee that all
 * jobs are finished.
 */
class CIndependentComputationEngine : public CSGObject
{
public:
	/** default constructor */
	CIndependentComputationEngine()
	: CSGObject()
	{
		SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** destructor */
	virtual ~CIndependentComputationEngine()
	{
		SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
	}

	/**
	 * abstract method that submits the jobs to the engine
	 *
	 * @param job the job to be computed
	 */
	virtual void submit_job(CIndependentJob* job) = 0;

	/** abstract method that blocks until all the jobs are completed */
	virtual void wait_for_all() = 0;

	/** @return object name */
	virtual const char* get_name() const
	{
		return "IndependentComputationEngine";
	}
};

}

#endif // INDEPENDENT_COMPUTATION_ENGINE_H_
