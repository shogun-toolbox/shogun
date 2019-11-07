/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
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
 *
 */

#include <gtest/gtest.h>
#include <shogun/lib/config.h>
#include <shogun/features/DenseSubSamplesFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGMatrix.h>

using namespace shogun;
TEST(DenseSubSamplesFeatures, test1)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolorance=1e-10;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);

	feat_train(0,0)=0.81263;
	feat_train(0,1)=0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=1.51752;
	feat_train(0,4)=1.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=0.5;
	feat_train(1,1)=0.4576;
	feat_train(1,2)=5.17637;
	feat_train(1,3)=2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=3.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=5.00000;

	auto features_train0=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto latent_features_train0=std::make_shared<DenseFeatures<float64_t>>(lat_feat_train);

	float64_t ell=0.5;
	auto kernel=std::make_shared<GaussianKernel>(10, 2*ell*ell);


	SGVector<int32_t> idx1(n);
	for(int i=0; i<idx1.vlen; i++)
		idx1[i]=i;

	auto features_train1=std::make_shared<DenseSubSamplesFeatures<float64_t>>(features_train0, idx1);

	SGVector<int32_t> idx2(m);
	for(int i=0; i<idx2.vlen; i++)
		idx2[i]=i;

	auto latent_features_train1=std::make_shared<DenseSubSamplesFeatures<float64_t>>(latent_features_train0, idx2);

	kernel->init(latent_features_train0, features_train0);
	SGMatrix<float64_t> res0=kernel->get_kernel_matrix();



	kernel->init(latent_features_train1, features_train1);
	SGMatrix<float64_t> res1=kernel->get_kernel_matrix();

	for(index_t i=0; i<res0.num_rows; i++)
	{
		for(index_t j=0; j<res0.num_cols; j++)
		{
			abs_tolorance = Math::get_abs_tolerance(res0(i,j), rel_tolorance);
			EXPECT_NEAR(res1(i,j), res0(i,j), abs_tolorance);
		}
	}






}

TEST(DenseSubSamplesFeatures, test2)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolorance=1e-10;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);

	feat_train(0,0)=0.81263;
	feat_train(0,1)=0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=1.51752;
	feat_train(0,4)=1.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=0.5;
	feat_train(1,1)=0.4576;
	feat_train(1,2)=5.17637;
	feat_train(1,3)=2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=3.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=5.00000;

	auto features_train0=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto latent_features_train0=std::make_shared<DenseFeatures<float64_t>>(lat_feat_train);

	SGVector<int32_t> idx1(n);
	for(int i=0; i<idx1.vlen; i++)
		idx1[i]=i;

	auto features_train1=std::make_shared<DenseSubSamplesFeatures<float64_t>>(features_train0, idx1);

	SGVector<int32_t> idx2(m);
	for(int i=0; i<idx2.vlen; i++)
		idx2[i]=i;

	auto latent_features_train1=std::make_shared<DenseSubSamplesFeatures<float64_t>>(latent_features_train0, idx2);

	for(int i=0; i<n; i++)
	{
		for(int j=0; j<m; j++)
		{
			float64_t res=latent_features_train0->dot(j,features_train0,i);

			abs_tolorance = Math::get_abs_tolerance(res, rel_tolorance);
			float64_t res1=latent_features_train1->dot(j,features_train0,i);
			EXPECT_NEAR(res1, res, abs_tolorance);
			float64_t res2=latent_features_train1->dot(j,features_train1,i);
			EXPECT_NEAR(res2, res, abs_tolorance);
		}
	}





}


TEST(DenseSubSamplesFeatures, test3)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolorance=1e-10;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);

	feat_train(0,0)=0.81263;
	feat_train(0,1)=0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=1.51752;
	feat_train(0,4)=1.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=0.5;
	feat_train(1,1)=0.4576;
	feat_train(1,2)=5.17637;
	feat_train(1,3)=2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=3.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=5.00000;

	auto features_train0=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto latent_features_train0=std::make_shared<DenseFeatures<float64_t>>(lat_feat_train);

	SGVector<int32_t> idx1(n);
	for(int i=0; i<idx1.vlen; i++)
		idx1[i]=n-i-1;

	auto features_train1=std::make_shared<DenseSubSamplesFeatures<float64_t>>(features_train0, idx1);

	SGVector<int32_t> idx2(m);
	for(int i=0; i<idx2.vlen; i++)
		idx2[i]=m-i-1;

	auto latent_features_train1=std::make_shared<DenseSubSamplesFeatures<float64_t>>(latent_features_train0, idx2);

	for(int i=0; i<n; i++)
	{
		for(int j=0; j<m; j++)
		{
			float64_t res=latent_features_train0->dot(j,features_train0,i);
			abs_tolorance = Math::get_abs_tolerance(res, rel_tolorance);

			float64_t res1=latent_features_train1->dot(m-j-1,features_train0,i);
			EXPECT_NEAR(res1, res, abs_tolorance);

			float64_t res2=latent_features_train1->dot(m-j-1,features_train1,n-i-1);
			EXPECT_NEAR(res2, res, abs_tolorance);
		}
	}





}

TEST(DenseSubSamplesFeatures, test5)
{
	index_t n=6;
	index_t dim=2;
	index_t m=4;
	float64_t rel_tolorance=1e-10;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);

	feat_train(0,0)=0.81263;
	feat_train(0,1)=0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=1.51752;
	feat_train(0,4)=1.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=0.5;
	feat_train(1,1)=0.4576;
	feat_train(1,2)=5.17637;
	feat_train(1,3)=2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=3.00000;
	lat_feat_train(0,2)=4.00000;
	lat_feat_train(0,3)=-8.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=5.00000;
	lat_feat_train(1,3)=-7.00000;

	auto features_train0=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto latent_features_train0=std::make_shared<DenseFeatures<float64_t>>(lat_feat_train);

	SGVector<int32_t> idx1(n/2);
	for(int i=0; i<n; i++)
	{
		if (i%2==0)
			idx1[i/2]=i;
	}

	auto features_train1=std::make_shared<DenseSubSamplesFeatures<float64_t>>(features_train0, idx1);

	SGVector<int32_t> idx2(m/2);
	for(int i=0; i<m; i++)
	{
		if (i%2==0)
			idx2[i/2]=i;
	}

	auto latent_features_train1=std::make_shared<DenseSubSamplesFeatures<float64_t>>(latent_features_train0, idx2);

	for(int i=0; i<n; i++)
	{
		for(int j=0; j<m; j++)
		{
			if (i%2==0 && j%2==0)
			{
				float64_t res=latent_features_train0->dot(j,features_train0,i);
				abs_tolorance = Math::get_abs_tolerance(res, rel_tolorance);

				float64_t res1=latent_features_train1->dot(j/2,features_train0,i);
				EXPECT_NEAR(res1, res, abs_tolorance);

				float64_t res2=latent_features_train1->dot(j/2,features_train1,i/2);
				EXPECT_NEAR(res2, res, abs_tolorance);
			}

		}
	}





}

TEST(DenseSubSamplesFeatures, test6)
{
	index_t n=6;
	index_t dim=2;
	index_t m=4;
	float64_t rel_tolorance=1e-10;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);

	feat_train(0,0)=0.81263;
	feat_train(0,1)=0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=1.51752;
	feat_train(0,4)=1.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=0.5;
	feat_train(1,1)=0.4576;
	feat_train(1,2)=5.17637;
	feat_train(1,3)=2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=3.00000;
	lat_feat_train(0,2)=4.00000;
	lat_feat_train(0,3)=-8.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=5.00000;
	lat_feat_train(1,3)=-7.00000;

	auto features_train0=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto latent_features_train0=std::make_shared<DenseFeatures<float64_t>>(lat_feat_train);

	SGVector<int32_t> idx1(n/2);
	for(int i=0; i<n; i++)
	{
		if (i%2==0)
			idx1[i/2]=i;
	}

	auto features_train1=std::make_shared<DenseSubSamplesFeatures<float64_t>>(features_train0, idx1);

	SGVector<int32_t> idx2(m/2);
	for(int i=0; i<m; i++)
	{
		if (i%2==0)
			idx2[i/2]=i;
	}

	auto latent_features_train1=std::make_shared<DenseSubSamplesFeatures<float64_t>>(latent_features_train0, idx2);


	SGMatrix<float64_t> feat_train2(dim, n/2);
	SGMatrix<float64_t> lat_feat_train2(dim, m/2);

	feat_train2(0,0)=0.81263;
	feat_train2(0,1)=1.17037;
	feat_train2(0,2)=1.57765;

	feat_train2(1,0)=0.5;
	feat_train2(1,1)=5.17637;
	feat_train2(1,2)=4.57765;

	lat_feat_train2(0,0)=1.00000;
	lat_feat_train2(0,1)=4.00000;

	lat_feat_train2(1,0)=3.00000;
	lat_feat_train2(1,1)=5.00000;

	auto features_train2=std::make_shared<DenseFeatures<float64_t>>(feat_train2);
	auto latent_features_train2=std::make_shared<DenseFeatures<float64_t>>(lat_feat_train2);

	float64_t ell=0.5;
	auto kernel=std::make_shared<GaussianKernel>(10, 2*ell*ell);



	kernel->init(latent_features_train2, features_train2);
	SGMatrix<float64_t> res2=kernel->get_kernel_matrix();



	kernel->init(latent_features_train1, features_train1);
	SGMatrix<float64_t> res1=kernel->get_kernel_matrix();

	for(index_t i=0; i<res1.num_rows; i++)
	{
		for(index_t j=0; j<res1.num_cols; j++)
		{
			abs_tolorance = Math::get_abs_tolerance(res2(i,j), rel_tolorance);
			EXPECT_NEAR(res1(i,j), res2(i,j), abs_tolorance);
		}
	}








}

TEST(DenseSubSamplesFeatures, test7)
{
	index_t n=6;
	index_t dim=2;
	index_t m=4;
	float64_t rel_tolorance=1e-10;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);

	feat_train(0,0)=0.81263;
	feat_train(0,1)=0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=1.51752;
	feat_train(0,4)=1.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=0.5;
	feat_train(1,1)=0.4576;
	feat_train(1,2)=5.17637;
	feat_train(1,3)=2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=3.00000;
	lat_feat_train(0,2)=4.00000;
	lat_feat_train(0,3)=-8.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=5.00000;
	lat_feat_train(1,3)=-7.00000;

	SGMatrix<float64_t> feat_train2(dim, n/2);
	feat_train2(0,0)=0.81263;
	feat_train2(0,1)=1.17037;
	feat_train2(0,2)=1.57765;

	feat_train2(1,0)=0.5;
	feat_train2(1,1)=5.17637;
	feat_train2(1,2)=4.57765;

	auto features_train0=std::make_shared<DenseFeatures<float64_t>>(feat_train2);
	auto latent_features_train0=std::make_shared<DenseFeatures<float64_t>>(lat_feat_train);

	SGVector<int32_t> idx1(n/2);
	for(int i=0; i<n; i++)
	{
		if (i%2==0)
			idx1[i/2]=i;
	}
	auto features_train1=std::make_shared<DenseFeatures<float64_t>>(feat_train);

	auto features_train2=std::make_shared<DenseSubSamplesFeatures<float64_t>>(features_train1, idx1);


	float64_t ell=0.5;
	auto kernel=std::make_shared<GaussianKernel>(10, 2*ell*ell);



	kernel->init(features_train0, latent_features_train0);
	SGMatrix<float64_t> res0=kernel->get_kernel_matrix();



	kernel->init(features_train2, latent_features_train0);
	SGMatrix<float64_t> res1=kernel->get_kernel_matrix();

	for(index_t i=0; i<res1.num_rows; i++)
	{
		for(index_t j=0; j<res1.num_cols; j++)
		{
			abs_tolorance = Math::get_abs_tolerance(res0(i,j), rel_tolorance);
			EXPECT_NEAR(res1(i,j), res0(i,j), abs_tolorance);
		}
	}







}
