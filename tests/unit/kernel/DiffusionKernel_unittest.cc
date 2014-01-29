/** The Shogun Machine Learning Toolbox
 *  Copyright (c) 2014, The Shogun-Team
 * All rights reserved.
 *
 * Distributed under the BSD 2-clause license:
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <shogun/kernel/DiffusionKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(DiffusionKernel, kernel_matrix)
{
	// X=[[1,2,2],[2,2,4],[2,4,2]]
	SGMatrix<float64_t> X(3,3);
	X(0,0)=1;
	X(0,1)=2;
	X(0,2)=2;
	X(1,0)=2;
	X(1,1)=2;
	X(1,2)=4;
	X(2,0)=2;
	X(2,1)=4;
	X(2,2)=2;

	// beta=[0.2,0.5,0.8]
	SGVector<float64_t> betas(3);
	betas[0]=0.2;
	betas[1]=0.5;
	betas[2]=0.8;

	// num_symbols=[5,3,4]
	SGVector<index_t> num_symbols(3);
	num_symbols[0]=5;
	num_symbols[1]=3;
	num_symbols[2]=4;

	CDenseFeatures<float64_t>* f=new CDenseFeatures<float64_t>(X);
	CDiffusionKernel* k = new CDiffusionKernel(betas, num_symbols);
	k->init(f, f);

	k->parallel->set_num_threads(1);

	SGMatrix<float64_t> K=k->get_kernel_matrix();

	EXPECT_NEAR(K(0,0), 1., 1e-7);
	EXPECT_NEAR(K(0,1), 0.21860429, 1e-7);
	EXPECT_NEAR(K(0,2), 0.13738457, 1e-7);
	EXPECT_NEAR(K(1,0), 0.21860429, 1e-7);
	EXPECT_NEAR(K(1,1), 1., 1e-7);
	EXPECT_NEAR(K(1,2), 0.45911797, 1e-7);
	EXPECT_NEAR(K(2,0), 0.13738457, 1e-7);
	EXPECT_NEAR(K(2,1), 0.45911797, 1e-7);
	EXPECT_NEAR(K(2,2), 1., 1e-7);

	SG_UNREF(k);
}
