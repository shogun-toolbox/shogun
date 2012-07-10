/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

void create_mean_data(SGMatrix<float64_t> target, float64_t difference)
{
	/* create data matrix for P and Q. P is a standard normal, Q is the same but
	 * has a mean difference in one dimension */
	for (index_t i=0; i<target.num_rows; ++i)
	{
		for (index_t j=0; j<target.num_cols/2; ++j)
			target(i,j)=CMath::randn_double();

		/* add mean difference in first dimension of second half of data */
		for (index_t j=target.num_cols/2; j<target.num_cols; ++j)
				target(i,j)=CMath::randn_double() + (i==0 ? difference : 0);
	}
}

SGMatrix<float64_t> create_fixed_data(index_t m, index_t dim)
{
	SGMatrix<float64_t> data(dim,2*m);
	for (index_t i=0; i<2*dim*m; ++i)
		data.matrix[i]=i*i;

	data.display_matrix("data");

	return data;
}

void test_linear_mmd_optimize_weights()
{
	index_t m=8;
	index_t dim=2;
	SGMatrix<float64_t> data=create_fixed_data(m, dim);

	/* create a number of kernels with different widths */
	SGVector<float64_t> sigmas(3);
	SGVector<float64_t> shogun_sigmas(sigmas.vlen);

	CCombinedKernel* kernel=new CCombinedKernel();
	CCombinedFeatures* features=new CCombinedFeatures();
	for (index_t i=0; i<sigmas.vlen; ++i)
	{
		sigmas[i]=CMath::pow(2.0, i-2)*1000;
		shogun_sigmas[i]=sigmas[i]*sigmas[i]*2;
		kernel->append_kernel(new CGaussianKernel(10, shogun_sigmas[i]));
		features->append_feature_obj(new CDenseFeatures<float64_t>(data));
	}

	sigmas.display_vector("sigmas");

	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, features, m);
	mmd->optimize_kernel_weights();

	kernel->get_subkernel_weights().display_vector("weights");

	SG_UNREF(mmd);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	sg_io->set_loglevel(MSG_DEBUG);

	test_linear_mmd_optimize_weights();

	exit_shogun();
	return 0;
}

