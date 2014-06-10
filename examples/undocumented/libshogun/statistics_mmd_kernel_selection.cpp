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
#include <shogun/statistics/QuadraticTimeMMD.h>
#include <shogun/statistics/MMDKernelSelectionCombOpt.h>
#include <shogun/statistics/MMDKernelSelectionCombMaxL2.h>
#include <shogun/statistics/MMDKernelSelectionOpt.h>
#include <shogun/statistics/MMDKernelSelectionMax.h>
#include <shogun/statistics/MMDKernelSelectionMedian.h>
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/features/streaming/generators/GaussianBlobsDataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

#ifdef HAVE_LAPACK
void kernel_choice_linear_time_mmd_opt_single()
{
	/* Note that the linear time mmd is designed for large datasets. Results on
	 * this small number will be bad (unstable, type I error wrong) */
	index_t m=1000;
	index_t num_blobs=3;
	float64_t distance=3;
	float64_t stretch=10;
	float64_t angle=CMath::PI/4;

	CGaussianBlobsDataGenerator* gen_p=new CGaussianBlobsDataGenerator(
				num_blobs, distance, stretch, angle);

	CGaussianBlobsDataGenerator* gen_q=new CGaussianBlobsDataGenerator(
				num_blobs, distance, 1, 1);

	/* create kernels */
	CCombinedKernel* combined=new CCombinedKernel();
	float64_t sigma_from=-3;
	float64_t sigma_to=10;
	float64_t sigma_step=1;
	float64_t sigma=sigma_from;
	while (sigma<=sigma_to)
	{
		/* shoguns kernel width is different */
		float64_t width=CMath::pow(2.0, sigma);
		float64_t sq_width_twice=width*width*2;
		combined->append_kernel(new CGaussianKernel(10, sq_width_twice));
		sigma+=sigma_step;
	}

	/* create MMD instance */
	CLinearTimeMMD* mmd=new CLinearTimeMMD(combined, gen_p, gen_q, m);

	/* kernel selection instance with regularisation term. May be replaced by
	 * other methods for selecting single kernels */
	CMMDKernelSelectionOpt* selection=
			new CMMDKernelSelectionOpt(mmd, 10E-5);
//
	/* select kernel that maximised MMD */
//	CMMDKernelSelectionMax* selection=
//			new CMMDKernelSelectionMax(mmd);

//	/* select kernel with width closest to median data distance */
//	CMMDKernelSelectionMedian* selection=
//			new CMMDKernelSelectionMedian(mmd, 10E-5);

	/* compute measures.
	 * For Opt: ratio of MMD and standard deviation
	 * For Max: MMDs of single kernels
	 * for Medigan: Does not work! */
	SG_SPRINT("computing ratios\n");
	SGVector<float64_t> ratios=selection->compute_measures();
	ratios.display_vector("ratios");

	/* select kernel using the maximum ratio (and cast) */
	SG_SPRINT("selecting kernel\n");
	CKernel* selected=selection->select_kernel();
	CGaussianKernel* casted=CGaussianKernel::obtain_from_generic(selected);
	SG_SPRINT("selected kernel width: %f\n", casted->get_width());
	mmd->set_kernel(selected);
	SG_UNREF(casted);
	SG_UNREF(selected);

	mmd->set_null_approximation_method(MMD1_GAUSSIAN);

	/* compute tpye I and II error (use many more trials). Type I error is only
	 * estimated to check MMD1_GAUSSIAN method for estimating the null
	 * distribution. Note that testing has to happen on difference data than
	 * kernel selecting, but the linear time mmd does this implicitly */
	float64_t alpha=0.05;
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


	SG_UNREF(selection);
}

void kernel_choice_linear_time_mmd_opt_comb()
{
	/* Note that the linear time mmd is designed for large datasets. Results on
	 * this small number will be bad (unstable, type I error wrong) */
	index_t m=1000;
	index_t num_blobs=3;
	float64_t distance=3;
	float64_t stretch=10;
	float64_t angle=CMath::PI/4;

	CGaussianBlobsDataGenerator* gen_p=new CGaussianBlobsDataGenerator(
				num_blobs, distance, stretch, angle);

	CGaussianBlobsDataGenerator* gen_q=new CGaussianBlobsDataGenerator(
				num_blobs, distance, 1, 1);

	/* create kernels */
	CCombinedKernel* combined=new CCombinedKernel();
	float64_t sigma_from=-3;
	float64_t sigma_to=10;
	float64_t sigma_step=1;
	float64_t sigma=sigma_from;
	index_t num_kernels=0;
	while (sigma<=sigma_to)
	{
		/* shoguns kernel width is different */
		float64_t width=CMath::pow(2.0, sigma);
		float64_t sq_width_twice=width*width*2;
		combined->append_kernel(new CGaussianKernel(10, sq_width_twice));
		sigma+=sigma_step;
		num_kernels++;
	}

	/* create MMD instance */
	CLinearTimeMMD* mmd=new CLinearTimeMMD(combined, gen_p, gen_q, m);

	/* kernel selection instance with regularisation term. May be replaced by
	 * other methods for selecting single kernels */
	CMMDKernelSelectionCombOpt* selection=
			new CMMDKernelSelectionCombOpt(mmd, 10E-5);

	/* maximise L2 regularised MMD */
//	CMMDKernelSelectionCombMaxL2* selection=
//			new CMMDKernelSelectionCombMaxL2(mmd, 10E-5);

	/* select kernel (does the same as above, but sets weights to kernel) */
	SG_SPRINT("selecting kernel\n");
	CKernel* selected=selection->select_kernel();
	CCombinedKernel* casted=CCombinedKernel::obtain_from_generic(selected);
	casted->get_subkernel_weights().display_vector("weights");
	mmd->set_kernel(selected);
	SG_UNREF(casted);
	SG_UNREF(selected);

	/* compute tpye I and II error (use many more trials). Type I error is only
	 * estimated to check MMD1_GAUSSIAN method for estimating the null
	 * distribution. Note that testing has to happen on difference data than
	 * kernel selecting, but the linear time mmd does this implicitly */
	mmd->set_null_approximation_method(MMD1_GAUSSIAN);
	float64_t alpha=0.05;
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


	SG_UNREF(selection);
}
#endif // HAVE_LAPACK

int main(int argc, char** argv)
{
	init_shogun_with_defaults();
//	sg_io->set_loglevel(MSG_DEBUG);

#ifdef HAVE_LAPACK
	/* select a single kernel for linear time MMD */
	kernel_choice_linear_time_mmd_opt_single();

	/* select combined kernels for linear time MMD */
	kernel_choice_linear_time_mmd_opt_comb();
#endif

	exit_shogun();
	return 0;
}

