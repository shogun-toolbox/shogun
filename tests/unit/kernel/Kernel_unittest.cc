/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
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

#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(Kernel, sum_symmetric_block_no_diag)
{
	const index_t num_feats=10;
	const index_t dim=3;

	// create random data
	CMath::init_random(100);
	SGMatrix<float64_t> data(dim, num_feats);
	for (index_t i=0; i<num_feats; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data(j, i)=CMath::randn_double();
	}
	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	// initialize a Gaussian kernel of width 1
	CGaussianKernel* kernel=new CGaussianKernel(feats, feats, 2);
	float64_t sum=kernel->sum_symmetric_block(0, num_feats);

	// check with the kernel matrix explicitely
	SGMatrix<float64_t> km=kernel->get_kernel_matrix();
	float64_t km_sum=0.0;
	for (index_t i=0; i<km.num_rows; i++)
	{
		for (index_t j=0; j<km.num_cols; ++j)
			km_sum+=i==j? 0 : km(i, j);
	}

	EXPECT_NEAR(sum, km_sum, 1E-13);

	// cleanup
	SG_UNREF(kernel);
}

TEST(Kernel, sum_symmetric_block_with_diag)
{
	const index_t num_feats=10;
	const index_t dim=3;

	// create random data
	CMath::init_random(100);
	SGMatrix<float64_t> data(dim, num_feats);
	for (index_t i=0; i<num_feats; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data(j, i)=CMath::randn_double();
	}
	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	// initialize a Gaussian kernel of width 1
	CGaussianKernel* kernel=new CGaussianKernel(feats, feats, 2);
	float64_t sum=kernel->sum_symmetric_block(0, num_feats, false);

	// check with the kernel matrix explicitely
	SGMatrix<float64_t> km=kernel->get_kernel_matrix();
	float64_t km_sum=0.0;
	for (index_t i=0; i<km.num_rows; i++)
	{
		for (index_t j=0; j<km.num_cols; ++j)
			km_sum+=km(i, j);
	}

	EXPECT_NEAR(sum, km_sum, 1E-13);

	// cleanup
	SG_UNREF(kernel);
}

TEST(Kernel, sum_block_with_diag)
{
	const index_t num_feats_p=10;
	const index_t num_feats_q=20;
	const index_t dim=3;

	// create random data
	CMath::init_random(100);

	SGMatrix<float64_t> data_p(dim, num_feats_p);
	for (index_t i=0; i<num_feats_p; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data_p(j, i)=CMath::randn_double();
	}

	SGMatrix<float64_t> data_q(dim, num_feats_q);
	for (index_t i=0; i<num_feats_q; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data_q(j, i)=CMath::randn_double();
	}

	CDenseFeatures<float64_t>* feats_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* feats_q=new CDenseFeatures<float64_t>(data_q);

	// initialize a Gaussian kernel of width 1
	CGaussianKernel* kernel=new CGaussianKernel(feats_p, feats_q, 2);
	float64_t sum=kernel->sum_block(0, 0, num_feats_p, num_feats_q);

	// check with the kernel rows and cols explicitly
	SGMatrix<float64_t> km=kernel->get_kernel_matrix();
	float64_t km_sum=0.0;
	for (index_t i=0; i<km.num_rows; i++)
	{
		for (index_t j=0; j<km.num_cols; ++j)
			km_sum+=km(i, j);
	}

	EXPECT_NEAR(sum, km_sum, 1E-13);

	// cleanup
	SG_UNREF(kernel);
}

TEST(Kernel, sum_block_no_diag)
{
	const index_t num_feats_p=10;
	const index_t num_feats_q=10;
	const index_t dim=3;

	// create random data
	CMath::init_random(100);

	SGMatrix<float64_t> data_p(dim, num_feats_p);
	for (index_t i=0; i<num_feats_p; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data_p(j, i)=CMath::randn_double();
	}

	SGMatrix<float64_t> data_q(dim, num_feats_q);
	for (index_t i=0; i<num_feats_q; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data_q(j, i)=CMath::randn_double();
	}

	CDenseFeatures<float64_t>* feats_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* feats_q=new CDenseFeatures<float64_t>(data_q);

	// initialize a Gaussian kernel of width 1
	CGaussianKernel* kernel=new CGaussianKernel(feats_p, feats_q, 2);
	float64_t sum=kernel->sum_block(0, 0, num_feats_p, num_feats_q, true);

	// check with the kernel rows and cols explicitly
	SGMatrix<float64_t> km=kernel->get_kernel_matrix();
	float64_t km_sum=0.0;
	for (index_t i=0; i<km.num_rows; i++)
	{
		for (index_t j=0; j<km.num_cols; ++j)
			km_sum+=i==j ? 0 : km(i, j);
	}

	EXPECT_NEAR(sum, km_sum, 1E-13);

	// cleanup
	SG_UNREF(kernel);
}

TEST(Kernel, row_wise_sum_symmetric_block_no_diag)
{
	const index_t num_feats=10;
	const index_t dim=3;

	// create random data
	CMath::init_random(100);
	SGMatrix<float64_t> data(dim, num_feats);
	for (index_t i=0; i<num_feats; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data(j, i)=CMath::randn_double();
	}
	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	// initialize a Gaussian kernel of width 1
	CGaussianKernel* kernel=new CGaussianKernel(feats, feats, 2);
	SGVector<float64_t> row_wise_sum_vec=kernel->row_wise_sum_symmetric_block(0,
			num_feats);

	// check with the kernel matrix explicitely
	SGMatrix<float64_t> km=kernel->get_kernel_matrix();
	for (index_t i=0; i<km.num_rows; i++)
	{
		float64_t row_wise_sum=0.0;
		for (index_t j=0; j<km.num_cols; ++j)
			row_wise_sum+=i==j? 0 : km(i, j);
		EXPECT_NEAR(row_wise_sum_vec[i], row_wise_sum, 1E-15);
	}

	// cleanup
	SG_UNREF(kernel);
}

TEST(Kernel, row_wise_sum_symmetric_block_with_diag)
{
	const index_t num_feats=10;
	const index_t dim=3;

	// create random data
	CMath::init_random(100);
	SGMatrix<float64_t> data(dim, num_feats);
	for (index_t i=0; i<num_feats; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data(j, i)=CMath::randn_double();
	}
	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	// initialize a Gaussian kernel of width 1
	CGaussianKernel* kernel=new CGaussianKernel(feats, feats, 2);
	SGVector<float64_t> row_wise_sum_vec=kernel->row_wise_sum_symmetric_block(0,
			num_feats, false);

	// check with the kernel matrix explicitely
	SGMatrix<float64_t> km=kernel->get_kernel_matrix();
	for (index_t i=0; i<km.num_rows; i++)
	{
		float64_t row_wise_sum=0.0;
		for (index_t j=0; j<km.num_cols; ++j)
			row_wise_sum+=km(i, j);
		EXPECT_NEAR(row_wise_sum_vec[i], row_wise_sum, 1E-15);
	}

	// cleanup
	SG_UNREF(kernel);
}

TEST(Kernel, row_wise_sum_squared_sum_symmetric_block_no_diag)
{
	const index_t num_feats=10;
	const index_t dim=3;

	// create random data
	CMath::init_random(100);
	SGMatrix<float64_t> data(dim, num_feats);
	for (index_t i=0; i<num_feats; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data(j, i)=CMath::randn_double();
	}
	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	// initialize a Gaussian kernel of width 1
	CGaussianKernel* kernel=new CGaussianKernel(feats, feats, 2);
	SGMatrix<float64_t> row_wise_sum_mat=
		kernel->row_wise_sum_squared_sum_symmetric_block(0, num_feats);

	// check with the kernel matrix explicitely
	SGMatrix<float64_t> km=kernel->get_kernel_matrix();
	for (index_t i=0; i<km.num_rows; i++)
	{
		float64_t row_wise_sum=0.0;
		float64_t row_wise_squared_sum=0.0;
		for (index_t j=0; j<km.num_cols; ++j)
		{
			float64_t k=km(i, j);
			row_wise_sum+=i==j? 0 : k;
			row_wise_squared_sum+=i==j? 0 : k*k;
		}
		EXPECT_NEAR(row_wise_sum_mat(i, 0), row_wise_sum, 1E-15);
		EXPECT_NEAR(row_wise_sum_mat(i, 1), row_wise_squared_sum, 1E-15);
	}

	// cleanup
	SG_UNREF(kernel);
}

TEST(Kernel, row_wise_sum_squared_sum_symmetric_block_with_diag)
{
	const index_t num_feats=10;
	const index_t dim=3;

	// create random data
	CMath::init_random(100);
	SGMatrix<float64_t> data(dim, num_feats);
	for (index_t i=0; i<num_feats; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data(j, i)=CMath::randn_double();
	}
	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	// initialize a Gaussian kernel of width 1
	CGaussianKernel* kernel=new CGaussianKernel(feats, feats, 2);
	SGMatrix<float64_t> row_wise_sum_mat=
		kernel->row_wise_sum_squared_sum_symmetric_block(0, num_feats, false);

	// check with the kernel matrix explicitely
	SGMatrix<float64_t> km=kernel->get_kernel_matrix();
	for (index_t i=0; i<km.num_rows; i++)
	{
		float64_t row_wise_sum=0.0;
		float64_t row_wise_squared_sum=0.0;
		for (index_t j=0; j<km.num_cols; ++j)
		{
			float64_t k=km(i, j);
			row_wise_sum+=k;
			row_wise_squared_sum+=k*k;
		}
		EXPECT_NEAR(row_wise_sum_mat(i, 0), row_wise_sum, 1E-15);
		EXPECT_NEAR(row_wise_sum_mat(i, 1), row_wise_squared_sum, 1E-15);
	}

	// cleanup
	SG_UNREF(kernel);
}

TEST(Kernel, row_col_wise_sum_block_with_diag)
{
	const index_t num_feats_p=10;
	const index_t num_feats_q=20;
	const index_t dim=3;

	// create random data
	CMath::init_random(100);

	SGMatrix<float64_t> data_p(dim, num_feats_p);
	for (index_t i=0; i<num_feats_p; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data_p(j, i)=CMath::randn_double();
	}

	SGMatrix<float64_t> data_q(dim, num_feats_q);
	for (index_t i=0; i<num_feats_q; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data_q(j, i)=CMath::randn_double();
	}

	CDenseFeatures<float64_t>* feats_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* feats_q=new CDenseFeatures<float64_t>(data_q);

	// initialize a Gaussian kernel of width 1
	CGaussianKernel* kernel=new CGaussianKernel(feats_p, feats_q, 2);
	SGVector<float64_t> row_col_wise_sum=kernel->row_col_wise_sum_block(0, 0,
			num_feats_p, num_feats_q);

	// check with the kernel rows and cols explicitly
	SGMatrix<float64_t> km=kernel->get_kernel_matrix();
	for (index_t i=0; i<km.num_rows; i++)
	{
		float64_t row_wise_sum=0;
		for (index_t j=0; j<km.num_cols; ++j)
			row_wise_sum+=km(i, j);
		EXPECT_NEAR(row_wise_sum, row_col_wise_sum[i], 1E-15);
	}

	for (index_t i=0; i<km.num_cols; i++)
	{
		float64_t col_wise_sum=0;
		for (index_t j=0; j<km.num_rows; ++j)
			col_wise_sum+=km(j, i);
		EXPECT_NEAR(col_wise_sum, row_col_wise_sum[i+num_feats_p], 1E-15);
	}

	// cleanup
	SG_UNREF(kernel);
}

TEST(Kernel, row_col_wise_sum_block_no_diag)
{
	const index_t num_feats_p=10;
	const index_t num_feats_q=10;
	const index_t dim=3;

	// create random data
	CMath::init_random(100);

	SGMatrix<float64_t> data_p(dim, num_feats_p);
	for (index_t i=0; i<num_feats_p; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data_p(j, i)=CMath::randn_double();
	}

	SGMatrix<float64_t> data_q(dim, num_feats_q);
	for (index_t i=0; i<num_feats_q; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			data_q(j, i)=CMath::randn_double();
	}

	CDenseFeatures<float64_t>* feats_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* feats_q=new CDenseFeatures<float64_t>(data_q);

	// initialize a Gaussian kernel of width 1
	CGaussianKernel* kernel=new CGaussianKernel(feats_p, feats_q, 2);
	SGVector<float64_t> row_col_wise_sum=kernel->row_col_wise_sum_block(0, 0,
			num_feats_p, num_feats_q, true);

	// check with the kernel rows and cols explicitly
	SGMatrix<float64_t> km=kernel->get_kernel_matrix();
	for (index_t i=0; i<km.num_rows; i++)
	{
		float64_t row_wise_sum=0;
		for (index_t j=0; j<km.num_cols; ++j)
			row_wise_sum+=i==j ? 0 : km(i, j);
		EXPECT_NEAR(row_wise_sum, row_col_wise_sum[i], 1E-15);
	}

	for (index_t i=0; i<km.num_cols; i++)
	{
		float64_t col_wise_sum=0;
		for (index_t j=0; j<km.num_rows; ++j)
			col_wise_sum+=i==j ? 0 :km(j, i);
		EXPECT_NEAR(col_wise_sum, row_col_wise_sum[i+num_feats_p], 1E-15);
	}

	// cleanup
	SG_UNREF(kernel);
}
