/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#ifndef SERIAL_COMPUTATION_ENGINE_H_
#define SERIAL_COMPUTATION_ENGINE_H_

#include <lib/config.h>
#include <lib/computation/engine/IndependentComputationEngine.h>

namespace shogun
{

/** @brief Class that computes multiple independent instances of
 * computation jobs sequentially
 */
class CSerialComputationEngine : public CIndependentComputationEngine
{
public:
	/** default constructor */
	CSerialComputationEngine();

	/** destructor */
	virtual ~CSerialComputationEngine();

	/**
	 * method that calls the job's compute method in each call, therefore
	 * blocks until the its computation is done
	 *
	 * @param job the job to be computed
	 */
	virtual void submit_job(CIndependentJob* job);

	/**
	 * method that returns when all the jobs computed, in this case it does
	 * nothing
	 */
	virtual void wait_for_all();

	/** @return object name */
	virtual const char* get_name() const
	{
		return "SerialComputationEngine";
	}
};

}

#endif // SERIAL_COMPUTATION_ENGINE_H_
