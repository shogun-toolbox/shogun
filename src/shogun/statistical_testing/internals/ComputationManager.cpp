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
	kernel_matrices.resize(n);
}

SGMatrix<float64_t>& ComputationManager::data(index_t i)
{
	return kernel_matrices[i];
}

void ComputationManager::enqueue_job(std::function<float64_t(SGMatrix<float64_t>)> job)
{
	jobq.push_back(job);
}

void ComputationManager::compute()
{
	for(auto job = jobq.begin(); job != jobq.end(); ++job)
	{
		const auto& operation = *job;
		std::vector<float64_t> results;
		if (gpu)
		{
			// TODO results = operation.compute_using_gpu(kernel_matrices);
		}
		else
		{
			results.resize(kernel_matrices.size());
#pragma omp parallel for
			for (size_t i = 0; i < kernel_matrices.size(); ++i)
			{
				results[i] = operation(kernel_matrices[i]);
			}
		}
		resultq.push(results);
	}
}

void ComputationManager::done()
{
	jobq.resize(0);
}

std::vector<float64_t> ComputationManager::next_result()
{
	std::vector<float64_t> result;
	if (!resultq.empty())
	{
		result = resultq.front();
		resultq.pop();
	}
	return result;
}

ComputationManager& ComputationManager::use_gpu()
{
	gpu = true;
	return *this;
}

ComputationManager& ComputationManager::use_cpu()
{
	gpu = false;
	return *this;
}
