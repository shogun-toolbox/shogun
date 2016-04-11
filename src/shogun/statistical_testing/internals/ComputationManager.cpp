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
#include <shogun/statistical_testing/internals/mmd/UnbiasedFull.h>

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

SGMatrix<float64_t>& ComputationManager::data(index_t i)
{
	return data_array[i];
}

void ComputationManager::enqueue_job(std::function<float64_t(SGMatrix<float64_t>)> job)
{
	job_array.push_back(job);
}

void ComputationManager::compute_data_parallel_jobs()
{
	result_array.resize(job_array.size());
	for(size_t j=0; j<job_array.size(); ++j)
	{
		const auto& compute_job=job_array[j];
		std::vector<float64_t> current_job_results(data_array.size());
		if (gpu)
		{
			// TODO current_job_results = compute_job.compute_using_gpu(data_array);
		}
		else
		{
#pragma omp parallel for
			for (size_t i=0; i<data_array.size(); ++i)
				current_job_results[i]=compute_job(data_array[i]);
		}
		result_array[j]=current_job_results;
	}
}

void ComputationManager::compute_task_parallel_jobs()
{
	result_array.resize(job_array.size());
#pragma omp parallel for
	for(size_t j=0; j<job_array.size(); ++j)
	{
		const auto& compute_job=job_array[j];
		std::vector<float64_t> current_job_results(data_array.size());
		if (gpu)
		{
			// TODO current_job_results = compute_job.compute_using_gpu(data_array);
		}
		else
		{
			for (size_t i=0; i<data_array.size(); ++i)
				current_job_results[i]=compute_job(data_array[i]);
		}
		result_array[j]=current_job_results;
	}
}

void ComputationManager::done()
{
	job_array.resize(0);
	result_array.resize(0);
}

std::vector<float64_t>& ComputationManager::result(index_t i)
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
