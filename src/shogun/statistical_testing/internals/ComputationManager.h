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

#ifndef COMPUTATION_MANAGER_H__
#define COMPUTATION_MANAGER_H__

#include <vector>
#include <functional>
#include <shogun/lib/common.h>

namespace shogun
{

template <typename T> class SGMatrix;

namespace internal
{

class ComputationManager
{
public:
	ComputationManager();
	~ComputationManager();

	void num_data(index_t n);
	SGMatrix<float64_t>& data(index_t i);

	void enqueue_job(std::function<float64_t(SGMatrix<float64_t>)> job);
	void compute_data_parallel_jobs();
	void compute_task_parallel_jobs();
	void done();
	std::vector<float64_t>& result(index_t i);

	ComputationManager& use_cpu();
	ComputationManager& use_gpu();
private:
	bool gpu;
	std::vector<SGMatrix<float64_t>> data_array;
	std::vector<std::function<float64_t(SGMatrix<float64_t>)>> job_array;
	std::vector<std::vector<float64_t>> result_array;
};

} // namespace internal

} // namespace shogun
#endif // COMPUTATION_MANAGER_H__
