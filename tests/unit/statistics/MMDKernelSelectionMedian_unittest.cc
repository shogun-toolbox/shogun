/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <base/init.h>
#include <statistics/QuadraticTimeMMD.h>
#include <statistics/MMDKernelSelectionMedian.h>
#include <features/streaming/StreamingFeatures.h>
#include <features/streaming/StreamingDenseFeatures.h>
#include <features/DenseFeatures.h>
#include <kernel/GaussianKernel.h>
#include <kernel/CombinedKernel.h>
#include <mathematics/Statistics.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(MMDKernelSelectionMedian,select_kernel)
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

	/* create Gaussian kernelkernels with sigmas 2^5 to 2^7 */
	CCombinedKernel* combined_kernel=new CCombinedKernel();
	//SG_SPRINT("adding widths (std)(shogun): ");
	for (index_t i=-5; i<=7; ++i)
	{
		/* shoguns kernel width is different */
		float64_t sigma=CMath::pow(2.0, i);
		float64_t sq_sigma_twice=sigma*sigma*2;
		//SG_SPRINT("(%f)(%f) ", sigma, sq_sigma_twice);
		combined_kernel->append_kernel(new CGaussianKernel(10, sq_sigma_twice));
	}
	//SG_SPRINT("\n");

	/* create MMD instance, convienience constructor */
	CQuadraticTimeMMD* mmd=new CQuadraticTimeMMD(combined_kernel, features_p,
			features_q);

	/* kernel selection instance */
	CMMDKernelSelectionMedian* selection=
			new CMMDKernelSelectionMedian(mmd);

	/* we know that a Gaussian kernel is returned when using median, the
	 * fifth one here one here */
	CGaussianKernel* kernel=(CGaussianKernel*)selection->select_kernel();
	//SG_SPRINT("median kernel width: %f\n", kernel->get_width());
	EXPECT_EQ(kernel->get_width(), 0.5);

	SG_UNREF(kernel);

	/* since convienience constructor was use for mmd, features have to be
	 * cleaned up by hand */
	SG_UNREF(features_p);
	SG_UNREF(features_q);

	SG_UNREF(selection);
}
