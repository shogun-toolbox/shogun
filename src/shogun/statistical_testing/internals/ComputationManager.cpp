/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2016  Soumyajit De
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http:/www.gnu.org/licenses/>.
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
		for (size_t i=0; i<data_array.size(); ++i)
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
