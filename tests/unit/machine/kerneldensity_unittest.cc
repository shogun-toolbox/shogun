/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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

#include <shogun/lib/SGVector.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/distributions/KernelDensity.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(KernelDensity,gaussian_kernel_with_euclidean_distance)
{
	SGMatrix<float64_t> data(2,4);
	data(0,0)=0;
	data(0,1)=0;
	data(0,2)=2;
	data(0,3)=2;
	data(1,0)=0;
	data(1,1)=2;
	data(1,2)=0;
	data(1,3)=2;

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);


	CKernelDensity* k=new CKernelDensity();
	k->train(feats);

	SGMatrix<float64_t> test(2,5);
	test(0,0)=1;
	test(1,0)=1;
	test(0,1)=0;
	test(1,1)=1;
	test(0,2)=2;
	test(1,2)=1;
	test(0,3)=1;
	test(1,3)=2;
	test(0,4)=1;
	test(1,4)=0;

	CDenseFeatures<float64_t>* testfeats=new CDenseFeatures<float64_t>(test);
	SGVector<float64_t> res=k->get_log_density(testfeats);

	EXPECT_NEAR(res[0],-2.83787706,1e-8);
	EXPECT_NEAR(res[1],-2.90409623,1e-8);
	EXPECT_NEAR(res[2],-2.90409623,1e-8);
	EXPECT_NEAR(res[3],-2.90409623,1e-8);
	EXPECT_NEAR(res[4],-2.90409623,1e-8);

	SG_UNREF(testfeats);
	SG_UNREF(feats);
	SG_UNREF(k);
}

TEST(KernelDensity,gaussian_kernel_with_manhattan_distance)
{
	SGMatrix<float64_t> data(2,4);
	data(0,0)=0;
	data(0,1)=0;
	data(0,2)=2;
	data(0,3)=2;
	data(1,0)=0;
	data(1,1)=2;
	data(1,2)=0;
	data(1,3)=2;

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);


	CKernelDensity* k=new CKernelDensity(1.0, K_GAUSSIAN, D_MANHATTAN);
	k->train(feats);

	SGMatrix<float64_t> test(2,5);
	test(0,0)=1;
	test(1,0)=1;
	test(0,1)=0;
	test(1,1)=1;
	test(0,2)=2;
	test(1,2)=1;
	test(0,3)=1;
	test(1,3)=2;
	test(0,4)=1;
	test(1,4)=0;

	CDenseFeatures<float64_t>* testfeats=new CDenseFeatures<float64_t>(test);
	SGVector<float64_t> res=k->get_log_density(testfeats);

	EXPECT_NEAR(res[0],-3.83787706,1e-8);
	EXPECT_NEAR(res[1],-3.01287431,1e-8);
	EXPECT_NEAR(res[2],-3.01287431,1e-8);
	EXPECT_NEAR(res[3],-3.01287431,1e-8);
	EXPECT_NEAR(res[4],-3.01287431,1e-8);

	SG_UNREF(testfeats);
	SG_UNREF(feats);
	SG_UNREF(k);
}