/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
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
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace internal;

TEST(KernelManager, store_precompute_restore)
{
	const index_t dim=1;
	const index_t num_vec=1;
	const index_t num_kernels=1;

	SGMatrix<float64_t> data_p(dim, num_vec);
	data_p(0, 0)=0;

	auto feats=new CDenseFeatures<float64_t>(data_p);
	auto kernel=new CGaussianKernel();
	kernel->set_width(0.5);

	KernelManager kernel_mgr(num_kernels);
	const KernelManager& const_kernel_mgr=kernel_mgr;

	kernel_mgr.kernel_at(0)=kernel;
	ASSERT_TRUE(const_kernel_mgr.kernel_at(0)->get_kernel_type()==K_GAUSSIAN);

	CKernel* k=const_kernel_mgr.kernel_at(0);
	k->init(feats, feats);
	kernel_mgr.precompute_kernel_at(0);
	ASSERT_TRUE(const_kernel_mgr.kernel_at(0)!=kernel);
	ASSERT_TRUE(const_kernel_mgr.kernel_at(0)->get_kernel_type()==K_CUSTOM);

	kernel_mgr.restore_kernel_at(0);
	ASSERT_TRUE(const_kernel_mgr.kernel_at(0)==kernel);
	ASSERT_TRUE(const_kernel_mgr.kernel_at(0)->get_kernel_type()==K_GAUSSIAN);
}
