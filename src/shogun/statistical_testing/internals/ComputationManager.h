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
#ifndef DOXYGEN_SHOULD_SKIP_THIS
class ComputationManager
{
public:
	ComputationManager();
	~ComputationManager();

	void num_data(index_t n);
	SGMatrix<float32_t>& data(index_t i);

	void enqueue_job(std::function<float32_t(SGMatrix<float32_t>)> job);
	void compute_data_parallel_jobs();
	void compute_task_parallel_jobs();
	void done();
	std::vector<float32_t>& result(index_t i);

	ComputationManager& use_cpu();
	ComputationManager& use_gpu();
private:
	bool gpu;
	std::vector<SGMatrix<float32_t> > data_array;
	std::vector<std::function<float32_t(const SGMatrix<float32_t>&)> > job_array;
	std::vector<std::vector<float32_t> > result_array;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace internal

} // namespace shogun
#endif // COMPUTATION_MANAGER_H__
