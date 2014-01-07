/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <statistics/MMDKernelSelectionMedian.h>
#include <statistics/LinearTimeMMD.h>
#include <features/streaming/StreamingFeatures.h>
#include <statistics/QuadraticTimeMMD.h>
#include <distance/EuclideanDistance.h>
#include <kernel/GaussianKernel.h>
#include <kernel/CombinedKernel.h>
#include <mathematics/Statistics.h>


using namespace shogun;

CMMDKernelSelectionMedian::CMMDKernelSelectionMedian() :
		CMMDKernelSelection()
{
	init();
}

CMMDKernelSelectionMedian::CMMDKernelSelectionMedian(
		CKernelTwoSampleTestStatistic* mmd, index_t num_data_distance) :
		CMMDKernelSelection(mmd)
{
	/* assert that a combined kernel is used */
	CKernel* kernel=mmd->get_kernel();
	CFeatures* lhs=kernel->get_lhs();
	CFeatures* rhs=kernel->get_rhs();
	REQUIRE(kernel, "%s::%s(): No kernel set!\n", get_name(), get_name());
	REQUIRE(kernel->get_kernel_type()==K_COMBINED, "%s::%s(): Requires "
			"CombinedKernel as kernel. Yours is %s", get_name(), get_name(),
			kernel->get_name());

	/* assert that all subkernels are Gaussian kernels */
	CCombinedKernel* combined=(CCombinedKernel*)kernel;

	for (index_t k_idx=0; k_idx<combined->get_num_kernels(); k_idx++)
	{
		CKernel* subkernel=combined->get_kernel(k_idx);
		REQUIRE(kernel, "%s::%s(): Subkernel (%d) of current kernel is not"
				" of type GaussianKernel\n", get_name(), get_name(), k_idx);
		SG_UNREF(subkernel);
	}

	/* assert 64 bit dense features since EuclideanDistance can only handle
	 * those */
	if (m_mmd->get_statistic_type()==S_QUADRATIC_TIME_MMD)
	{
		CFeatures* features=((CQuadraticTimeMMD*)m_mmd)->get_p_and_q();
		REQUIRE(features->get_feature_class()==C_DENSE &&
				features->get_feature_type()==F_DREAL, "%s::select_kernel(): "
				"Only 64 bit float dense features allowed, these are \"%s\""
				" and of type %d\n",
				get_name(), features->get_name(), features->get_feature_type());
		SG_UNREF(features);
	}
	else if (m_mmd->get_statistic_type()==S_LINEAR_TIME_MMD)
	{
		CStreamingFeatures* p=((CLinearTimeMMD*)m_mmd)->get_streaming_p();
		CStreamingFeatures* q=((CLinearTimeMMD*)m_mmd)->get_streaming_q();
		REQUIRE(p->get_feature_class()==C_STREAMING_DENSE &&
				p->get_feature_type()==F_DREAL, "%s::select_kernel(): "
				"Only 64 bit float streaming dense features allowed, these (p) "
				"are \"%s\" and of type %d\n",
				get_name(), p->get_name(), p->get_feature_type());

		REQUIRE(p->get_feature_class()==C_STREAMING_DENSE &&
				p->get_feature_type()==F_DREAL, "%s::select_kernel(): "
				"Only 64 bit float streaming dense features allowed, these (q) "
				"are \"%s\" and of type %d\n",
				get_name(), q->get_name(), q->get_feature_type());
		SG_UNREF(p);
		SG_UNREF(q);
	}

	SG_UNREF(kernel);
	SG_UNREF(lhs);
	SG_UNREF(rhs);

	init();

	m_num_data_distance=num_data_distance;
}

CMMDKernelSelectionMedian::~CMMDKernelSelectionMedian()
{
}

void CMMDKernelSelectionMedian::init()
{
	SG_ADD(&m_num_data_distance, "m_num_data_distance", "Number of elements to "
			"to compute median distance on", MS_NOT_AVAILABLE);

	/* this is a sensible value */
	m_num_data_distance=1000;
}

SGVector<float64_t> CMMDKernelSelectionMedian::compute_measures()
{
	SG_ERROR("%s::compute_measures(): Not implemented. Use select_kernel() "
			"method!\n", get_name());
	return SGVector<float64_t>();
}

CKernel* CMMDKernelSelectionMedian::select_kernel()
{
	/* number of data for distace */
	index_t num_data=CMath::min(m_num_data_distance, m_mmd->get_m());

	SGMatrix<float64_t> dists;

	/* compute all pairwise distances, depends which mmd statistic is used */
	if (m_mmd->get_statistic_type()==S_QUADRATIC_TIME_MMD)
	{
		/* fixed data, create merged copy of a random subset */

		/* create vector with that correspond to the num_data first points of
		 * each distribution, remember data is stored jointly */
		SGVector<index_t> subset(num_data*2);
		index_t m=m_mmd->get_m();
		for (index_t i=0; i<num_data; ++i)
		{
			/* num_data samples from each half of joint sample */
			subset[i]=i;
			subset[i+num_data]=i+m;
		}

		/* add subset and compute pairwise distances */
		CQuadraticTimeMMD* quad_mmd=(CQuadraticTimeMMD*)m_mmd;
		CFeatures* features=quad_mmd->get_p_and_q();
		features->add_subset(subset);

		/* cast is safe, see constructor */
		CDenseFeatures<float64_t>* dense_features=
				(CDenseFeatures<float64_t>*) features;

		CEuclideanDistance* distance=new CEuclideanDistance(dense_features,
				dense_features);
		dists=distance->get_distance_matrix();
		features->remove_subset();
		SG_UNREF(distance);
		SG_UNREF(features);
	}
	else if (m_mmd->get_statistic_type()==S_LINEAR_TIME_MMD)
	{
		/* just stream the desired number of points */
		CLinearTimeMMD* linear_mmd=(CLinearTimeMMD*)m_mmd;

		CStreamingFeatures* p=linear_mmd->get_streaming_p();
		CStreamingFeatures* q=linear_mmd->get_streaming_q();

		/* cast is safe, see constructor */
		CDenseFeatures<float64_t>* p_streamed=(CDenseFeatures<float64_t>*)
				p->get_streamed_features(num_data);
		CDenseFeatures<float64_t>* q_streamed=(CDenseFeatures<float64_t>*)
					q->get_streamed_features(num_data);

		/* for safety */
		SG_REF(p_streamed);
		SG_REF(q_streamed);

		/* create merged feature object */
		CDenseFeatures<float64_t>* merged=(CDenseFeatures<float64_t>*)
				p_streamed->create_merged_copy(q_streamed);

		/* compute pairwise distances */
		CEuclideanDistance* distance=new CEuclideanDistance(merged, merged);
		dists=distance->get_distance_matrix();

		/* clean up */
		SG_UNREF(distance);
		SG_UNREF(p_streamed);
		SG_UNREF(q_streamed);
		SG_UNREF(p);
		SG_UNREF(q);
	}

	/* create a vector where the zeros have been removed, use upper triangle
	 * only since distances are symmetric */
	SGVector<float64_t> dist_vec(dists.num_rows*(dists.num_rows-1)/2);
	index_t write_idx=0;
	for (index_t i=0; i<dists.num_rows; ++i)
	{
		for (index_t j=i+1; j<dists.num_rows; ++j)
			dist_vec[write_idx++]=dists(i,j);
	}

	/* now we have distance matrix, compute median, allow to modify matrix */
	float64_t median_distance=CStatistics::median(dist_vec, true);
	SG_DEBUG("median_distance: %f\n", median_distance);

	/* shogun has no square and factor two in its kernel width, MATLAB does
	 * median_width = sqrt(0.5*median_distance), we do this */
	float64_t shogun_sigma=median_distance;
	SG_DEBUG("kernel width (shogun): %f\n", shogun_sigma);

	/* now of all kernels, find the one which has its width closest
	 * Cast is safe due to constructor of MMDKernelSelection class */
	CCombinedKernel* combined=(CCombinedKernel*)m_mmd->get_kernel();
	float64_t min_distance=CMath::MAX_REAL_NUMBER;
	CKernel* min_kernel=NULL;
	float64_t distance;
	for (index_t i=0; i<combined->get_num_subkernels(); ++i)
	{
		CKernel* current=combined->get_kernel(i);
		REQUIRE(current->get_kernel_type()==K_GAUSSIAN, "%s::select_kernel(): "
				"%d-th kernel is not a Gaussian but \"%s\"!\n", get_name(), i,
				current->get_name());

		/* check if width is closer to median width */
		distance=CMath::abs(((CGaussianKernel*)current)->get_width()-
				shogun_sigma);

		if (distance<min_distance)
		{
			min_distance=distance;
			min_kernel=current;
		}

		/* next kernel */
		SG_UNREF(current);
	}
	SG_UNREF(combined);

	/* returned referenced kernel */
	SG_REF(min_kernel);
	return min_kernel;
}
