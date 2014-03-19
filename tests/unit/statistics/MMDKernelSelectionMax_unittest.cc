/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/statistics/QuadraticTimeMMD.h>
#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/statistics/MMDKernelSelectionMax.h>
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(MMDKernelSelectionMax,select_kernel_quadratic_time_mmd)
{
	index_t m=8;
	index_t d=3;
	SGMatrix<float64_t> data(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	/* normalise data to get some reasonable values for Q matrix */
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	//SG_SPRINT("%f, %f\n", max_p, max_q);

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	//data_p.display_matrix("data_p");
	//data_q.display_matrix("data_q");

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* create kernels with sigmas 2^5 to 2^7 */
	CCombinedKernel* combined_kernel=new CCombinedKernel();
	for (index_t i=5; i<=7; ++i)
	{
		/* shoguns kernel width is different */
		float64_t sigma=CMath::pow(2, i);
		float64_t sq_sigma_twice=sigma*sigma*2;
		combined_kernel->append_kernel(new CGaussianKernel(10, sq_sigma_twice));
	}

	/* create MMD instance, convienience constructor */
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(combined_kernel, features_p,
			features_q);

	/* kernel selection instance */
	CMMDKernelSelectionMax* selection=
			new CMMDKernelSelectionMax(mmd);

	/* assert correct mmd values, maxmmd criterion is already checked with
	 * linear time mmd maxmmd selection. Do biased and unbiased m*MMD */

	/* unbiased m*MMD */
	SGVector<float64_t> measures=selection->compute_measures();
	//measures.display_vector("unbiased mmd");
//	unbiased_quad_mmds =
//	   0.001164382204818   0.000291185913881   0.000072802127661
	EXPECT_LE(CMath::abs(measures[0]-0.001164382204818), 10E-15);
	EXPECT_LE(CMath::abs(measures[1]-0.000291185913881), 10E-15);
	EXPECT_LE(CMath::abs(measures[2]-0.000072802127661), 10E-15);

	/* biased m*MMD */
	mmd->set_statistic_type(BIASED);
	measures=selection->compute_measures();
	//measures.display_vector("biased mmd");
//	biased_quad_mmds =
//	   0.001534961982492   0.000383849322208   0.000095969134022
	EXPECT_LE(CMath::abs(measures[0]-0.001534961982492), 10E-15);
	EXPECT_LE(CMath::abs(measures[1]-0.000383849322208), 10E-15);
	EXPECT_LE(CMath::abs(measures[2]-0.000095969134022), 10E-15);

	/* since convienience constructor was use for mmd, features have to be
	 * cleaned up by hand */
	SG_UNREF(features_p);
	SG_UNREF(features_q);

	SG_UNREF(selection);
}

TEST(MMDKernelSelectionMax,select_kernel_linear_time_mmd)
{
	index_t m=8;
	index_t d=3;
	SGMatrix<float64_t> data(d,2*m);
	for (index_t i=0; i<2*d*m; ++i)
		data.matrix[i]=i;

	/* create data matrix for each features (appended is not supported) */
	SGMatrix<float64_t> data_p(d, m);
	memcpy(&(data_p.matrix[0]), &(data.matrix[0]), sizeof(float64_t)*d*m);

	SGMatrix<float64_t> data_q(d, m);
	memcpy(&(data_q.matrix[0]), &(data.matrix[d*m]), sizeof(float64_t)*d*m);

	/* normalise data to get some reasonable values for Q matrix */
	float64_t max_p=data_p.max_single();
	float64_t max_q=data_q.max_single();

	//SG_SPRINT("%f, %f\n", max_p, max_q);

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	//data_p.display_matrix("data_p");
	//data_q.display_matrix("data_q");

	CDenseFeatures<float64_t>* features_p=new CDenseFeatures<float64_t>(data_p);
	CDenseFeatures<float64_t>* features_q=new CDenseFeatures<float64_t>(data_q);

	/* create stremaing features from dense features */
	CStreamingFeatures* streaming_p=
			new CStreamingDenseFeatures<float64_t>(features_p);
	CStreamingFeatures* streaming_q=
			new CStreamingDenseFeatures<float64_t>(features_q);

	/* create kernels with sigmas 2^5 to 2^7 */
	CCombinedKernel* combined_kernel=new CCombinedKernel();
	for (index_t i=5; i<=7; ++i)
	{
		/* shoguns kernel width is different */
		float64_t sigma=CMath::pow(2, i);
		float64_t sq_sigma_twice=sigma*sigma*2;
		combined_kernel->append_kernel(new CGaussianKernel(10, sq_sigma_twice));
	}

	/* create MMD instance */
	CLinearTimeMMD* mmd=new CLinearTimeMMD(combined_kernel, streaming_p,
			streaming_q, m);

	/* kernel selection instance */
	CMMDKernelSelectionMax* selection=
			new CMMDKernelSelectionMax(mmd);

	/* start streaming features parser */
	streaming_p->start_parser();
	streaming_q->start_parser();

	/* assert that the correct kernel is returned since I checked the MMD
	 * already very often */
	CKernel* result=selection->select_kernel();
	CGaussianKernel* casted=dynamic_cast<CGaussianKernel*>(result);
	ASSERT(casted);

	/* assert weights against matlab */
	CKernel* reference=combined_kernel->get_first_kernel();
	ASSERT(result==reference);
	SG_UNREF(reference);

	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(selection);
	SG_UNREF(result);
}
