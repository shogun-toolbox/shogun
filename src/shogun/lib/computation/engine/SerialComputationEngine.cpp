/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/common.h>
#include <shogun/lib/computation/job/IndependentJob.h>
#include <shogun/lib/computation/engine/SerialComputationEngine.h>
#include <shogun/lib/computation/engine/IndependentComputationEngine.h>
#include <shogun/io/SGIO.h>

namespace shogun
{

CSerialComputationEngine::CSerialComputationEngine()
	: CIndependentComputationEngine()
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CSerialComputationEngine::~CSerialComputationEngine()
{
	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

void CSerialComputationEngine::submit_job(CIndependentJob* job)
{
	SG_DEBUG("Entering. The job is being computed!\n");

	REQUIRE(job, "Job to be computed is NULL\n");
	job->compute();

	SG_DEBUG("The job is computed. Leaving!\n");
}

void CSerialComputationEngine::wait_for_all()
{
	SG_DEBUG("All jobs are computed!\n");
}

}
