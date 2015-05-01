/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
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

#include <shogun/lib/config.h>

#ifdef HAVE_LINALG_LIB
#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianARDKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;


TEST(GaussianARDKernel_scalar,get_kernel_matrix)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolerance=1e-10;
	float64_t abs_tolerance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);

	float64_t ell=2.0;
	CLinearARDKernel* kernel=new CGaussianARDKernel(10, 2*ell*ell);
	float64_t weight=0.5;
	kernel->set_scalar_weights(weight);

	float64_t ell2=ell/weight;
	CGaussianKernel* kernel2=new CGaussianKernel(10, 2*ell2*ell2);

	SG_REF(features_train)
	SG_REF(latent_features_train)

	kernel->init(features_train, latent_features_train);
	kernel2->init(features_train, latent_features_train);

	SGMatrix<float64_t> mat=kernel->get_kernel_matrix();
	SGMatrix<float64_t> mat2=kernel2->get_kernel_matrix();
	for(int32_t i=0;i<mat.num_rows;i++)
	{
		for(int32_t j=0;j<mat.num_cols;j++)
		{
			abs_tolerance=CMath::get_abs_tolerance(mat2(i,j),rel_tolerance);
			EXPECT_NEAR(mat(i,j),mat2(i,j),abs_tolerance);
		}
	}

	kernel->init(features_train, features_train);

	kernel2->init(features_train, features_train);

	mat=kernel->get_kernel_matrix();
	mat2=kernel2->get_kernel_matrix();
	for(int32_t i=0;i<mat.num_rows;i++)
	{
		for(int32_t j=0;j<mat.num_cols;j++)
		{
			abs_tolerance=CMath::get_abs_tolerance(mat2(i,j),rel_tolerance);
			EXPECT_NEAR(mat(i,j),mat2(i,j),abs_tolerance);
		}
	}

	// cleanup
	SG_UNREF(kernel);
	SG_UNREF(kernel2);
	SG_UNREF(features_train)
	SG_UNREF(latent_features_train)
}

TEST(GaussianARDKernel_scalar,get_parameter_gradient)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolerance=1e-10;
	float64_t abs_tolerance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);

	float64_t ell=4.0;
	CLinearARDKernel* kernel=new CGaussianARDKernel(10, 2*ell*ell);
	float64_t weight=1.0;
	kernel->set_scalar_weights(weight);

	float64_t ell2=ell/weight;
	CGaussianKernel* kernel2=new CGaussianKernel(10, 2*ell2*ell2);


	SG_REF(features_train)
	SG_REF(latent_features_train)

	kernel->init(features_train, latent_features_train);
	kernel2->init(features_train, latent_features_train);

	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("width");
	TParameter* width_param2=kernel2->m_gradient_parameters->get_parameter("width");

	SGMatrix<float64_t> mat=kernel->get_parameter_gradient(width_param);
	SGMatrix<float64_t> mat2=kernel2->get_parameter_gradient(width_param2);
	for(int32_t i=0;i<mat.num_rows;i++)
	{
		for(int32_t j=0;j<mat.num_cols;j++)
		{
			abs_tolerance=CMath::get_abs_tolerance(mat2(i,j),rel_tolerance);
			EXPECT_NEAR(mat(i,j),mat2(i,j),abs_tolerance);
		}
	}

	kernel->init(features_train, features_train);
	kernel2->init(features_train, features_train);

	mat=kernel->get_parameter_gradient(width_param);
	mat2=kernel2->get_parameter_gradient(width_param2);
	for(int32_t i=0;i<mat.num_rows;i++)
	{
		for(int32_t j=0;j<mat.num_cols;j++)
		{
			abs_tolerance=CMath::get_abs_tolerance(mat2(i,j),rel_tolerance);
			EXPECT_NEAR(mat(i,j),mat2(i,j),abs_tolerance);
		}
	}

	// cleanup
	SG_UNREF(kernel);
	SG_UNREF(kernel2);
	SG_UNREF(features_train)
	SG_UNREF(latent_features_train)
}


TEST(GaussianARDKernel_vector,get_kernel_matrix)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolerance=1e-10;
	float64_t abs_tolerance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1;
	lat_feat_train(0,1)=23;
	lat_feat_train(0,2)=4;

	lat_feat_train(1,0)=3;
	lat_feat_train(1,1)=2;
	lat_feat_train(1,2)=-5;

	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);


	float64_t ell=1.0;
	CLinearARDKernel* kernel=new CGaussianARDKernel(10, 2*ell*ell);

	float64_t weight1=6.0;
	float64_t weight2=3.0;
	SGVector<float64_t> weights(dim);
	weights[0]=1.0/weight1;
	weights[1]=1.0/weight2;
	kernel->set_vector_weights(weights);

	SG_REF(latent_features_train)
	SG_REF(features_train)
	kernel->init(features_train, latent_features_train);

	//result from GPML 3.5
	//0.483748918128241   0.000268463258484   0.235348892827356
	//0.676321622589282   0.000172691024978   0.001624066485753
	//0.379307967154712   0.000301388413582   0.000236847574321
	//0.163638175047594   0.000074274235677   0.471639781078544
	//0.392276838870544   0.038462421551295   0.004574637837373
	//0.889607966835858   0.006010671270288   0.031352436711192

	SGMatrix<float64_t> mat=kernel->get_kernel_matrix();
	abs_tolerance = CMath::get_abs_tolerance(0.483748918128241, rel_tolerance);
	EXPECT_NEAR(mat(0,0),  0.483748918128241,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000268463258484, rel_tolerance);
	EXPECT_NEAR(mat(0,1),  0.000268463258484,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.235348892827356, rel_tolerance);
	EXPECT_NEAR(mat(0,2),  0.235348892827356,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.676321622589282, rel_tolerance);
	EXPECT_NEAR(mat(1,0),  0.676321622589282,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000172691024978, rel_tolerance);
	EXPECT_NEAR(mat(1,1),  0.000172691024978,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.001624066485753, rel_tolerance);
	EXPECT_NEAR(mat(1,2),  0.001624066485753,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.379307967154712, rel_tolerance);
	EXPECT_NEAR(mat(2,0),  0.379307967154712,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000301388413582, rel_tolerance);
	EXPECT_NEAR(mat(2,1),  0.000301388413582,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000236847574321, rel_tolerance);
	EXPECT_NEAR(mat(2,2),  0.000236847574321,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.163638175047594, rel_tolerance);
	EXPECT_NEAR(mat(3,0),  0.163638175047594,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000074274235677, rel_tolerance);
	EXPECT_NEAR(mat(3,1),  0.000074274235677,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.471639781078544, rel_tolerance);
	EXPECT_NEAR(mat(3,2),  0.471639781078544,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.392276838870544, rel_tolerance);
	EXPECT_NEAR(mat(4,0),  0.392276838870544,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.038462421551295, rel_tolerance);
	EXPECT_NEAR(mat(4,1),  0.038462421551295,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.004574637837373, rel_tolerance);
	EXPECT_NEAR(mat(4,2),  0.004574637837373,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.889607966835858, rel_tolerance);
	EXPECT_NEAR(mat(5,0),  0.889607966835858,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.006010671270288, rel_tolerance);
	EXPECT_NEAR(mat(5,1),  0.006010671270288,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.031352436711192, rel_tolerance);
	EXPECT_NEAR(mat(5,2),  0.031352436711192,  abs_tolerance);


	kernel->init(latent_features_train, latent_features_train);
	//result from GPML 3.5
	//1.000000000000000   0.001138802761346   0.025208965963144
	//0.001138802761346   1.000000000000000   0.000436766814255
	//0.025208965963144   0.000436766814255   1.000000000000000

	mat=kernel->get_kernel_matrix();
	abs_tolerance = CMath::get_abs_tolerance(1.000000000000000, rel_tolerance);
	EXPECT_NEAR(mat(0,0),  1.000000000000000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.001138802761346, rel_tolerance);
	EXPECT_NEAR(mat(0,1),  0.001138802761346,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.025208965963144, rel_tolerance);
	EXPECT_NEAR(mat(0,2),  0.025208965963144,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.001138802761346, rel_tolerance);
	EXPECT_NEAR(mat(1,0),  0.001138802761346,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.000000000000000, rel_tolerance);
	EXPECT_NEAR(mat(1,1),  1.000000000000000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000436766814255, rel_tolerance);
	EXPECT_NEAR(mat(1,2),  0.000436766814255,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.025208965963144, rel_tolerance);
	EXPECT_NEAR(mat(2,0),  0.025208965963144,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000436766814255, rel_tolerance);
	EXPECT_NEAR(mat(2,1),  0.000436766814255,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.000000000000000, rel_tolerance);
	EXPECT_NEAR(mat(2,2),  1.000000000000000,  abs_tolerance);

	// cleanup
	SG_UNREF(kernel);
	SG_UNREF(latent_features_train)
	SG_UNREF(features_train)
}

TEST(GaussianARDKernel_matrix,get_kernel_matrix1)
{
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolerance=1e-10;
	float64_t abs_tolerance;

	SGMatrix<float64_t> lat_feat_train(dim, m);
	SGMatrix<float64_t> lat_feat_train2(dim, m);

	lat_feat_train(0,0)=1;
	lat_feat_train(0,1)=23;
	lat_feat_train(0,2)=4;

	lat_feat_train(1,0)=3;
	lat_feat_train(1,1)=2;
	lat_feat_train(1,2)=-5;

	lat_feat_train2(0,0)=1;
	lat_feat_train2(0,1)=23;
	lat_feat_train2(0,2)=4;

	lat_feat_train2(1,0)=3;
	lat_feat_train2(1,1)=2;
	lat_feat_train2(1,2)=-5;

	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CDenseFeatures<float64_t>* latent_features_train2=new CDenseFeatures<float64_t>(lat_feat_train2);

	float64_t ell=1.0;
	CLinearARDKernel* kernel=new CGaussianARDKernel(10, 2*ell*ell);

	int32_t t_dim=2;
	SGMatrix<float64_t> weights(t_dim,dim);
	//the weights is a upper triangular matrix since GPML 3.5 only supports this type
	float64_t weight1=0.02;
	float64_t weight2=-0.4;
	float64_t weight3=0;
	float64_t weight4=0.01;
	weights(0,0)=weight1;
	weights(0,1)=weight2;
	weights(1,0)=weight3;
	weights(1,1)=weight4;
	kernel->set_matrix_weights(weights);

	SG_REF(latent_features_train)
	SG_REF(latent_features_train2)

	kernel->init(latent_features_train, latent_features_train2);
	//result from GPML 3.5
	//1.000000000000000   0.702682587860637   0.004907454025841
	//0.702682587860637   1.000000000000000   0.053362341348083
	//0.004907454025841   0.053362341348083   1.000000000000000
	SGMatrix<float64_t> mat=kernel->get_kernel_matrix();
	abs_tolerance = CMath::get_abs_tolerance(1.000000000000000, rel_tolerance);
	EXPECT_NEAR(mat(0,0),  1.000000000000000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.702682587860637, rel_tolerance);
	EXPECT_NEAR(mat(0,1),  0.702682587860637,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.004907454025841, rel_tolerance);
	EXPECT_NEAR(mat(0,2),  0.004907454025841,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.702682587860637, rel_tolerance);
	EXPECT_NEAR(mat(1,0),  0.702682587860637,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.000000000000000, rel_tolerance);
	EXPECT_NEAR(mat(1,1),  1.000000000000000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.053362341348083, rel_tolerance);
	EXPECT_NEAR(mat(1,2),  0.053362341348083,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.004907454025841, rel_tolerance);
	EXPECT_NEAR(mat(2,0),  0.004907454025841,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.053362341348083, rel_tolerance);
	EXPECT_NEAR(mat(2,1),  0.053362341348083,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.000000000000000, rel_tolerance);
	EXPECT_NEAR(mat(2,2),  1.000000000000000,  abs_tolerance);

	// cleanup
	SG_UNREF(kernel);
	SG_UNREF(latent_features_train)
	SG_UNREF(latent_features_train2)
}


TEST(GaussianARDKernel_matrix,get_kernel_matrix2)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolerance=1e-10;
	float64_t abs_tolerance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1;
	lat_feat_train(0,1)=23;
	lat_feat_train(0,2)=4;

	lat_feat_train(1,0)=3;
	lat_feat_train(1,1)=2;
	lat_feat_train(1,2)=-5;

	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);


	float64_t ell=1.0;
	CLinearARDKernel* kernel=new CGaussianARDKernel(10, 2*ell*ell);

	int32_t t_dim=2;
	SGMatrix<float64_t> weights(t_dim,dim);
	//the weights is a upper triangular matrix since GPML 3.5 only supports this type
	float64_t weight1=0.02;
	float64_t weight2=-0.4;
	float64_t weight3=0;
	float64_t weight4=0.01;
	weights(0,0)=weight1;
	weights(0,1)=weight2;
	weights(1,0)=weight3;
	weights(1,1)=weight4;
	kernel->set_matrix_weights(weights);

	SG_REF(latent_features_train)
	SG_REF(features_train)
	kernel->init(features_train, latent_features_train);

	//result from GPML 3.5
	//0.394350178890907   0.871562091809855   0.165480906152021
	//0.592382640849672   0.176215460003241   0.000103321679752
	//0.248938103583867   0.043100670101141   0.000005310787411
	//0.093436892436847   0.408865150874797   0.555930926275610
	//0.891287769081377   0.418582374577455   0.000915039915173
	//0.994999180183564   0.760525767334597   0.006768257533514

	SGMatrix<float64_t> mat=kernel->get_kernel_matrix();

	abs_tolerance = CMath::get_abs_tolerance(0.394350178890907, rel_tolerance);
	EXPECT_NEAR(mat(0,0),  0.394350178890907,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.871562091809855, rel_tolerance);
	EXPECT_NEAR(mat(0,1),  0.871562091809855,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.165480906152021, rel_tolerance);
	EXPECT_NEAR(mat(0,2),  0.165480906152021,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.592382640849672, rel_tolerance);
	EXPECT_NEAR(mat(1,0),  0.592382640849672,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.176215460003241, rel_tolerance);
	EXPECT_NEAR(mat(1,1),  0.176215460003241,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000103321679752, rel_tolerance);
	EXPECT_NEAR(mat(1,2),  0.000103321679752,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.248938103583867, rel_tolerance);
	EXPECT_NEAR(mat(2,0),  0.248938103583867,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.043100670101141, rel_tolerance);
	EXPECT_NEAR(mat(2,1),  0.043100670101141,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000005310787411, rel_tolerance);
	EXPECT_NEAR(mat(2,2),  0.000005310787411,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.093436892436847, rel_tolerance);
	EXPECT_NEAR(mat(3,0),  0.093436892436847,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.408865150874797, rel_tolerance);
	EXPECT_NEAR(mat(3,1),  0.408865150874797,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.555930926275610, rel_tolerance);
	EXPECT_NEAR(mat(3,2),  0.555930926275610,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.891287769081377, rel_tolerance);
	EXPECT_NEAR(mat(4,0),  0.891287769081377,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.418582374577455, rel_tolerance);
	EXPECT_NEAR(mat(4,1),  0.418582374577455,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000915039915173, rel_tolerance);
	EXPECT_NEAR(mat(4,2),  0.000915039915173,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.994999180183564, rel_tolerance);
	EXPECT_NEAR(mat(5,0),  0.994999180183564,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.760525767334597, rel_tolerance);
	EXPECT_NEAR(mat(5,1),  0.760525767334597,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.006768257533514, rel_tolerance);
	EXPECT_NEAR(mat(5,2),  0.006768257533514,  abs_tolerance);

	kernel->init(latent_features_train, latent_features_train);
	//result from GPML 3.5
	//1.000000000000000   0.702682587860637   0.004907454025841
	//0.702682587860637   1.000000000000000   0.053362341348083
	//0.004907454025841   0.053362341348083   1.000000000000000
	mat=kernel->get_kernel_matrix();
	abs_tolerance = CMath::get_abs_tolerance(1.000000000000000, rel_tolerance);
	EXPECT_NEAR(mat(0,0),  1.000000000000000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.702682587860637, rel_tolerance);
	EXPECT_NEAR(mat(0,1),  0.702682587860637,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.004907454025841, rel_tolerance);
	EXPECT_NEAR(mat(0,2),  0.004907454025841,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.702682587860637, rel_tolerance);
	EXPECT_NEAR(mat(1,0),  0.702682587860637,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.000000000000000, rel_tolerance);
	EXPECT_NEAR(mat(1,1),  1.000000000000000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.053362341348083, rel_tolerance);
	EXPECT_NEAR(mat(1,2),  0.053362341348083,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.004907454025841, rel_tolerance);
	EXPECT_NEAR(mat(2,0),  0.004907454025841,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.053362341348083, rel_tolerance);
	EXPECT_NEAR(mat(2,1),  0.053362341348083,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.000000000000000, rel_tolerance);
	EXPECT_NEAR(mat(2,2),  1.000000000000000,  abs_tolerance);

	// cleanup
	SG_UNREF(kernel);
	SG_UNREF(latent_features_train)
	SG_UNREF(features_train)
}

TEST(GaussianARDKernel,get_kernel_diagonal)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolerance=1e-10;
	float64_t abs_tolerance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);

	float64_t ell=0.5;
	CLinearARDKernel* kernel=new CGaussianARDKernel(10, 2*ell*ell);

	int32_t t_dim=2;
	SGMatrix<float64_t> weights(t_dim,dim);
	//the weights is a upper triangular matrix since GPML 3.5 only supports this type
	float64_t weight1=0.02;
	float64_t weight2=-0.4;
	float64_t weight3=0;
	float64_t weight4=0.01;
	weights(0,0)=weight1;
	weights(0,1)=weight2;
	weights(1,0)=weight3;
	weights(1,1)=weight4;
	kernel->set_matrix_weights(weights);

	SG_REF(features_train)
	SG_REF(latent_features_train)

	kernel->init(features_train, latent_features_train);

	SGVector<float64_t> vec=kernel->get_kernel_diagonal();
	SGMatrix<float64_t> mat2=kernel->get_kernel_matrix();
	SGVector<float64_t> vec2=mat2.get_diagonal_vector();

	for(int32_t i=0;i<vec.vlen;i++)
	{
		abs_tolerance=CMath::get_abs_tolerance(vec2[i],rel_tolerance);
		EXPECT_NEAR(vec[i],vec2[i],abs_tolerance);
	}

	kernel->init(features_train, features_train);

	vec=kernel->get_kernel_diagonal();
	mat2=kernel->get_kernel_matrix();
	vec2=mat2.get_diagonal_vector();

	for(int32_t i=0;i<vec.vlen;i++)
	{
		abs_tolerance=CMath::get_abs_tolerance(vec2[i],rel_tolerance);
		EXPECT_NEAR(vec[i],vec2[i],abs_tolerance);
	}

	// cleanup
	SG_UNREF(kernel);
	SG_UNREF(features_train)
	SG_UNREF(latent_features_train)
}

TEST(GaussianARDKernel,get_parameter_gradient_diagonal)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolerance=1e-10;
	float64_t abs_tolerance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);

	float64_t ell=4.0;
	CLinearARDKernel* kernel=new CGaussianARDKernel(10, 2*ell*ell);
	float64_t weight=1.0;
	kernel->set_scalar_weights(weight);

	float64_t ell2=ell/weight;
	CGaussianKernel* kernel2=new CGaussianKernel(10, 2*ell2*ell2);

	SG_REF(features_train)
	SG_REF(latent_features_train)

	kernel->init(features_train, latent_features_train);
	kernel2->init(features_train, latent_features_train);

	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("width");
	TParameter* width_param2=kernel2->m_gradient_parameters->get_parameter("width");

	SGVector<float64_t> vec=kernel->get_parameter_gradient_diagonal(width_param);
	SGVector<float64_t> vec2=kernel2->get_parameter_gradient_diagonal(width_param2);
	for(int32_t j=0;j<vec.vlen;j++)
	{
		abs_tolerance=CMath::get_abs_tolerance(vec2[j],rel_tolerance);
		EXPECT_NEAR(vec[j],vec2[j],abs_tolerance);
	}

	kernel->init(features_train, features_train);
	kernel2->init(features_train, features_train);

	vec=kernel->get_parameter_gradient_diagonal(width_param);
	vec2=kernel2->get_parameter_gradient_diagonal(width_param2);
	for(int32_t j=0;j<vec.vlen;j++)
	{
		abs_tolerance=CMath::get_abs_tolerance(vec2[j],rel_tolerance);
		EXPECT_NEAR(vec[j],vec2[j],abs_tolerance);
	}

	// cleanup
	SG_UNREF(kernel);
	SG_UNREF(kernel2);
	SG_UNREF(features_train)
	SG_UNREF(latent_features_train)
}

#endif /* HAVE_LINALG_LIB */
