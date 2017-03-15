/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 - 2017 Soumyajit De
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

#ifndef KERNEL_MANAGER_H__
#define KERNEL_MANAGER_H__

#include <vector>
#include <memory>
#include <shogun/lib/common.h>
#include <shogun/statistical_testing/internals/InitPerKernel.h>

namespace shogun
{

class CKernel;
class CDistance;
class CCustomDistance;
class CCustomKernel;

namespace internal
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
class KernelManager
{
public:
	KernelManager();
	explicit KernelManager(index_t num_kernels);
	~KernelManager();

	InitPerKernel kernel_at(index_t i);
	CKernel* kernel_at(index_t i) const;

	void push_back(CKernel* kernel);
	const index_t num_kernels() const;

	void precompute_kernel_at(index_t i);
	void restore_kernel_at(index_t i);

	void clear();
	bool same_distance_type() const;
	CDistance* get_distance_instance() const;
	void set_precomputed_distance(CCustomDistance* distance) const;
	void unset_precomputed_distance() const;
private:
	std::vector<std::shared_ptr<CKernel> > m_kernels;
	std::vector<std::shared_ptr<CCustomKernel> > m_precomputed_kernels;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS
}

}

#endif // KERNEL_MANAGER_H__
