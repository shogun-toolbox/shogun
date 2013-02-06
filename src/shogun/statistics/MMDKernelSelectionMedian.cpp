/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/statistics/MMDKernelSelectionMedian.h>
#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/statistics/QuadraticTimeMMD.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Statistics.h>


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
	/* assert that a Gaussian kernel is used */
	CKernel* kernel=mmd->get_kernel();
	CFeatures* lhs=kernel->get_lhs();
	CFeatures* rhs=kernel->get_rhs();
	REQUIRE(kernel, "%s::%s(): No kernel set!\n", get_name(), get_name())
	REQUIRE(kernel->get_kernel_type()==K_GAUSSIAN, "%s::%s(): Requires "
			"CGaussianKernel as kernel", get_name(), get_name());

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
	SG_WARNING("register params!\n")

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

		/* create vector with random indices subset */
		SGVector<index_t> indices(m_mmd->get_m());
		indices.range_fill();
		indices.permute();
		SGVector<index_t> subset(num_data);
		memcpy(subset.vector, indices.vector, num_data);

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
		SG_REF(p_streamed);
		SG_REF(q_streamed);
		SG_UNREF(p);
		SG_UNREF(q);
	}

	/* now we have distance matrix, compute median, allow to modify matrix */
	float64_t median_distance=CStatistics::matrix_median(dists, true);

	/* shogun has no square and factor two in its kernel width */
	float64_t shogun_sigma=CMath::pow(median_distance, 2);

	/* cast is safe, see constructor */
	CGaussianKernel* kernel=(CGaussianKernel*) m_mmd->get_kernel();
	kernel->set_width(shogun_sigma);

	/* no unref since kernel is returned */
	return kernel;
}
