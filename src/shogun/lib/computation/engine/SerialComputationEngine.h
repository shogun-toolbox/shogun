/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Thoralf Klein, Yuyu Zhang, Björn Esser
 */

#ifndef SERIAL_COMPUTATION_ENGINE_H_
#define SERIAL_COMPUTATION_ENGINE_H_

#include <shogun/lib/config.h>

#include <shogun/lib/computation/engine/IndependentComputationEngine.h>

namespace shogun
{
class CIndependentJob;

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
