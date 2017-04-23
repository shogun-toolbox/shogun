/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 - 2017 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/lib/SGMatrix.h>
#include <shogun/statistical_testing/internals/ComputationManager.h>

using namespace shogun;
using namespace internal;

ComputationManager::ComputationManager()
{
}

ComputationManager::~ComputationManager()
{
}

void ComputationManager::num_data(index_t n)
{
	data_array.resize(n);
}

SGMatrix<float32_t>& ComputationManager::data(index_t i)
{
	return data_array[i];
}

void ComputationManager::enqueue_job(std::function<float32_t(SGMatrix<float32_t>)> job)
{
	job_array.push_back(job);
}

void ComputationManager::compute_data_parallel_jobs()
{
	// this is used when there are more number of data blocks to be processed
	// than there are jobs
	result_array.resize(job_array.size());
	for (size_t j=0; j<job_array.size(); ++j)
		result_array[j].resize(data_array.size());

	if (gpu)
	{
		// TODO current_job_results = compute_job.compute_using_gpu(data_array);
	}
	else
	{
#pragma omp parallel for
		for (int64_t i=0; i<(int64_t)data_array.size(); ++i)
		{
			// using a temporary vector to hold the result, because it is
			// cache friendly, since the original result matrix would lead
			// to several cache misses, specially because the data is also
			// being used here
			std::vector<float32_t> current_data_results(job_array.size());
			for (size_t j=0; j<job_array.size(); ++j)
			{
				const auto& compute_job=job_array[j];
				current_data_results[j]=compute_job(data_array[i]);
			}
			// data is no more required, less cache miss when we just have to
			// store the results
			for (size_t j=0; j<current_data_results.size(); ++j)
				result_array[j][i]=current_data_results[j];
		}
	}
}

void ComputationManager::compute_task_parallel_jobs()
{
	// this is used when there are more number of jobs to be processed
	// than there are data blocks
	result_array.resize(job_array.size());
	for (size_t j=0; j<job_array.size(); ++j)
		result_array[j].resize(data_array.size());

	if (gpu)
	{
		// TODO current_job_results = compute_job.compute_using_gpu(data_array);
	}
	else
	{
		// TODO figure out other ways to deal with the parallelization in presence of
		// eigen3. presently due to that, using OpenMP here messes things up!
//#pragma omp parallel for
		for (size_t j=0; j<job_array.size(); ++j)
		{
			const auto& compute_job=job_array[j];
			// result_array[j][i] is contiguous, cache miss is minimized
			for (size_t i=0; i<data_array.size(); ++i)
				result_array[j][i]=compute_job(data_array[i]);
		}
	}
}

void ComputationManager::done()
{
	job_array.resize(0);
	result_array.resize(0);
}

std::vector<float32_t>& ComputationManager::result(index_t i)
{
	return result_array[i];
}

ComputationManager& ComputationManager::use_gpu()
{
	gpu=true;
	return *this;
}

ComputationManager& ComputationManager::use_cpu()
{
	gpu=false;
	return *this;
}
