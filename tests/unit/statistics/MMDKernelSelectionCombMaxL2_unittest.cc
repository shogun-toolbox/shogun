/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/statistics/MMDKernelSelectionCombMaxL2.h>
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(MMDKernelSelectionCombMaxL2, select_kernel)
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
	CMMDKernelSelectionCombMaxL2* selection=new CMMDKernelSelectionCombMaxL2(
			mmd);

	/* start streaming features parser */
	streaming_p->start_parser();
	streaming_q->start_parser();

	CKernel* result=selection->select_kernel();
	CCombinedKernel* casted=dynamic_cast<CCombinedKernel*>(result);
	ASSERT(casted);
	SGVector<float64_t> weights=casted->get_subkernel_weights();
	//weights.display_vector("weights");

	/* assert weights against local machine computed results */
//	previously w_l2 was =
//	   0.761798188424313
//	   0.190556119182660
//	   0.047645692393028
	EXPECT_NEAR(weights[0], 0.76188388214402547, 1E-10);
	EXPECT_NEAR(weights[1], 0.19049185019530737, 1E-10);
	EXPECT_NEAR(weights[2], 0.047624267660667088, 1E-10);

	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(selection);
	SG_UNREF(result);
}
