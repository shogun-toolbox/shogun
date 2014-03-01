/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

void linear_time_mmd()
{
	/* note that the linear time statistic is designed for much larger datasets
	 * so increase to get reasonable results */
	index_t m=1000;
	index_t dim=2;
	float64_t difference=0.5;

	/* streaming data generator for mean shift distributions */
	CMeanShiftDataGenerator* gen_p=new CMeanShiftDataGenerator(0, dim);
	CMeanShiftDataGenerator* gen_q=new CMeanShiftDataGenerator(difference, dim);

	/* set kernel a-priori. usually one would do some kernel selection. See
	 * other examples for this. */
	float64_t width=10;
	CGaussianKernel* kernel=new CGaussianKernel(10, width);

	/* create linear time mmd instance */
	index_t blocksize=1000;
	CLinearTimeMMD* mmd=new CLinearTimeMMD(kernel, gen_p, gen_q, m, blocksize);

	/* perform test: compute p-value and test if null-hypothesis is rejected for
	 * a test level of 0.05 */
	float64_t alpha=0.05;

	/* using bootstrapping (not reccomended for linear time MMD, since slow).
	 * Also, in practice, use at least 250 iterations */
	mmd->set_null_approximation_method(PERMUTATION);
	mmd->set_num_permutation_iterations(10);
	float64_t p_value_bootstrap=mmd->perform_test();
	/* reject if p-value is smaller than test level */
	SG_SPRINT("bootstrap: p!=q: %d\n", p_value_bootstrap<alpha);

	/* using Gaussian approximation (use large sample size, check type I error).
	 * Also, in practice, use at least 250 iterations */
	mmd->set_null_approximation_method(MMD1_GAUSSIAN);
	float64_t p_value_gaussian=mmd->perform_test();
	/* reject if p-value is smaller than test level */
	SG_SPRINT("gaussian approx: p!=q: %d\n", p_value_gaussian<alpha);

	/* compute tpye I and II error (use many more trials in practice).
	 * Type I error is only estimated to check MMD1_GAUSSIAN method for
	 * estimating the null distribution. Note that testing has to happen on
	 * difference data than kernel selection, but the linear time mmd does this
	 * implicitly and we used a fixed kernel here. */
	index_t num_trials=5;
	SGVector<float64_t> typeIerrors(num_trials);
	SGVector<float64_t> typeIIerrors(num_trials);
	for (index_t i=0; i<num_trials; ++i)
	{
		/* this effectively means that p=q - rejecting is tpye I error */
		mmd->set_simulate_h0(true);
		typeIerrors[i]=mmd->perform_test()>alpha;
		mmd->set_simulate_h0(false);

		typeIIerrors[i]=mmd->perform_test()>alpha;
	}

	SG_SPRINT("type I error: %f\n", CStatistics::mean(typeIerrors));
	SG_SPRINT("type II error: %f\n", CStatistics::mean(typeIIerrors));

	SG_UNREF(mmd);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();
//	sg_io->set_loglevel(MSG_DEBUG);

	linear_time_mmd();

	exit_shogun();
	return 0;
}

