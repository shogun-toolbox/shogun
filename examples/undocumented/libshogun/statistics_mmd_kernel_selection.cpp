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
#include <shogun/statistics/MMDKernelSelectionOptComb.h>
#include <shogun/statistics/MMDKernelSelectionOptSingle.h>
#include <shogun/statistics/MMDKernelSelectionMax.h>
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

void test_kernel_choice_linear_time_mmd_opt_comb()
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

	SG_SPRINT("%f, %f\n", max_p, max_q);

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	data_p.display_matrix("data_p");
	data_q.display_matrix("data_q");

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

	/* create MMD instance, no kernel, handled by kernel selection later */
	CLinearTimeMMD* mmd=new CLinearTimeMMD(combined_kernel, streaming_p,
			streaming_q, m);

	/* kernel selection instance with regularisation term */
	CMMDKernelSelectionOptComb* selection=new CMMDKernelSelectionOptComb(mmd,
			10E-5);

	/* start streaming features parser */
	streaming_p->start_parser();
	streaming_q->start_parser();

	CKernel* result=selection->select_kernel();
	CCombinedKernel* casted=dynamic_cast<CCombinedKernel*>(result);
	ASSERT(casted);
	SGVector<float64_t> weights=casted->get_subkernel_weights();
	weights.display_vector("weights");

	/* assert weights against matlab */
//	w_opt =
//	   0.761798190146441
//	   0.190556117891148
//	   0.047645691962411
	ASSERT(CMath::abs(weights[0]-0.761798190146441)<10E-15);
	ASSERT(CMath::abs(weights[1]-0.190556117891148)<10E-15);
	ASSERT(CMath::abs(weights[2]-0.047645691962411)<10E-15);


	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(selection);
	SG_UNREF(result);
}

void test_kernel_choice_linear_time_mmd_opt_single()
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

	SG_SPRINT("%f, %f\n", max_p, max_q);

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	data_p.display_matrix("data_p");
	data_q.display_matrix("data_q");

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

	/* create MMD instance, no kernel, handled by kernel selection later */
	CLinearTimeMMD* mmd=new CLinearTimeMMD(combined_kernel, streaming_p,
			streaming_q, m);

	/* kernel selection instance with regularisation term */
	CMMDKernelSelectionOptSingle* selection=
			new CMMDKernelSelectionOptSingle(mmd, 10E-5);

	/* start streaming features parser */
	streaming_p->start_parser();
	streaming_q->start_parser();

	SGVector<float64_t> ratios=selection->compute_measures();
	ratios.display_vector("ratios");

	/* assert weights against matlab */
//	ratios =
//	   0.947668253683719
//	   0.336041393822230
//	   0.093824478467851
	ASSERT(CMath::abs(ratios[0]-0.947668253683719)<10E-15);
	ASSERT(CMath::abs(ratios[1]-0.336041393822230)<10E-15);
	ASSERT(CMath::abs(ratios[2]-0.093824478467851)<10E-15);

	/* start streaming features parser */
	streaming_p->end_parser();
	streaming_q->end_parser();

	SG_UNREF(selection);
}

void test_kernel_choice_linear_time_mmd_maxmmd_single()
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

	SG_SPRINT("%f, %f\n", max_p, max_q);

	for (index_t i=0; i<d*m; ++i)
	{
		data_p.matrix[i]/=max_p;
		data_q.matrix[i]/=max_q;
	}

	data_p.display_matrix("data_p");
	data_q.display_matrix("data_q");

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

	/* create MMD instance, no kernel, handled by kernel selection later */
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

int main(int argc, char** argv)
{
	init_shogun_with_defaults();
//	sg_io->set_loglevel(MSG_DEBUG);

	test_kernel_choice_linear_time_mmd_opt_comb();
	test_kernel_choice_linear_time_mmd_opt_single();
	test_kernel_choice_linear_time_mmd_maxmmd_single();

	exit_shogun();
	return 0;
}

