/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/base/init.h>
#include <shogun/statistics/QuadraticTimeMMD.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

void quadratic_time_mmd()
{
	/* number of examples kept low in order to make things fast */
	index_t m=30;
	index_t dim=2;
	float64_t difference=0.5;

	/* streaming data generator for mean shift distributions */
	CMeanShiftDataGenerator* gen_p=new CMeanShiftDataGenerator(0, dim);
	CMeanShiftDataGenerator* gen_q=new CMeanShiftDataGenerator(difference, dim);

	/* stream some data from generator */
	CFeatures* feat_p=gen_p->get_streamed_features(m);
	CFeatures* feat_q=gen_q->get_streamed_features(m);

	/* set kernel a-priori. usually one would do some kernel selection. See
	 * other examples for this. */
	float64_t width=10;
	CGaussianKernel* kernel=new CGaussianKernel(10, width);

	/* create quadratic time mmd instance. Note that this constructor
	 * copies p and q and does not reference them */
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(kernel, feat_p, feat_q);

	/* perform test: compute p-value and test if null-hypothesis is rejected for
	 * a test level of 0.05 */
	float64_t alpha=0.05;

	/* using permutation (slow, not the most reliable way. Consider pre-
	 * computing the kernel when using it, see below).
	 * Also, in practice, use at least 250 iterations */
	mmd->set_null_approximation_method(PERMUTATION);
	mmd->set_num_permutation_iterations(3);
	float64_t p_value=mmd->perform_test();
	/* reject if p-value is smaller than test level */
	SG_SPRINT("bootstrap: p!=q: %d\n", p_value<alpha);

	/* using spectrum method. Use at least 250 samples from null.
	 * This is consistent but sometimes breaks, always monitor type I error.
	 * See tutorial for number of eigenvalues to use .
	 * Only works with BIASED statistic */
	mmd->set_statistic_type(BIASED);
	mmd->set_null_approximation_method(MMD2_SPECTRUM);
	mmd->set_num_eigenvalues_spectrum(3);
	mmd->set_num_samples_sepctrum(250);
	p_value=mmd->perform_test();
	/* reject if p-value is smaller than test level */
	SG_SPRINT("spectrum: p!=q: %d\n", p_value<alpha);

	/* using gamma method. This is a quick hack, which works most of the time
	 * but is NOT guaranteed to. See tutorial for details.
	 * Only works with BIASED statistic */
	mmd->set_statistic_type(BIASED);
	mmd->set_null_approximation_method(MMD2_GAMMA);
	p_value=mmd->perform_test();
	/* reject if p-value is smaller than test level */
	SG_SPRINT("gamma: p!=q: %d\n", p_value<alpha);

	/* compute tpye I and II error (use many more trials in practice).
	 * Type I error is not necessary if one uses permutation. We do it here
	 * anyway, but note that this is an efficient way of computing it.
	 * Also note that testing has to happen on
	 * difference data than kernel selection, but the linear time mmd does this
	 * implicitly and we used a fixed kernel here. */
	mmd->set_null_approximation_method(PERMUTATION);
	mmd->set_num_permutation_iterations(5);
	index_t num_trials=5;
	SGVector<float64_t> type_I_errors(num_trials);
	SGVector<float64_t> type_II_errors(num_trials);
	SGVector<index_t> inds(2*m);
	inds.range_fill();
	CFeatures* p_and_q=mmd->get_p_and_q();

	/* use a precomputed kernel to be faster */
	kernel->init(p_and_q, p_and_q);
	CCustomKernel* precomputed=new CCustomKernel(kernel);
	mmd->set_kernel(precomputed);
	for (index_t i=0; i<num_trials; ++i)
	{
		/* this effectively means that p=q - rejecting is tpye I error */
		inds.permute();
		precomputed->add_row_subset(inds);
		precomputed->add_col_subset(inds);
		type_I_errors[i]=mmd->perform_test()>alpha;
		precomputed->remove_row_subset();
		precomputed->remove_col_subset();

		/* on normal data, this gives type II error */
		type_II_errors[i]=mmd->perform_test()>alpha;
	}
	SG_UNREF(p_and_q);

	SG_SPRINT("type I error: %f\n", CStatistics::mean(type_I_errors));
	SG_SPRINT("type II error: %f\n", CStatistics::mean(type_II_errors));

	/* clean up */
	SG_UNREF(mmd);
	SG_UNREF(gen_p);
	SG_UNREF(gen_q);

	/* convienience constructor of MMD was used, these were not referenced */
	SG_UNREF(feat_p);
	SG_UNREF(feat_q);
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();
//	sg_io->set_loglevel(MSG_DEBUG);

	quadratic_time_mmd();

	exit_shogun();
	return 0;
}

