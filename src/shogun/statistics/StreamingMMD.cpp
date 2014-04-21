/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012-2013 Heiko Strathmann
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/statistics/StreamingMMD.h>
#include <shogun/features/Features.h>
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/lib/List.h>

using namespace shogun;

CStreamingMMD::CStreamingMMD() : CKernelTwoSampleTest()
{
	init();
}

CStreamingMMD::CStreamingMMD(CKernel* kernel, CStreamingFeatures* p,
		CStreamingFeatures* q, index_t m, index_t blocksize) :
		CKernelTwoSampleTest(kernel, NULL, m)
{
	init();

	m_streaming_p=p;
	SG_REF(m_streaming_p);

	m_streaming_q=q;
	SG_REF(m_streaming_q);

	m_blocksize=blocksize;
}

CStreamingMMD::~CStreamingMMD()
{
	SG_UNREF(m_streaming_p);
	SG_UNREF(m_streaming_q);

	/* m_kernel is SG_UNREFed in base desctructor */
}

void CStreamingMMD::init()
{
	SG_ADD((CSGObject**)&m_streaming_p, "streaming_p", "Streaming features p",
				MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_streaming_q, "streaming_q", "Streaming features p",
				MS_NOT_AVAILABLE);
	SG_ADD(&m_blocksize, "blocksize", "Number of elements processed at once",
				MS_NOT_AVAILABLE);
	SG_ADD(&m_simulate_h0, "simulate_h0", "Whether p and q are mixed",
				MS_NOT_AVAILABLE);

	m_streaming_p=NULL;
	m_streaming_q=NULL;
	m_blocksize=10000;
	m_simulate_h0=false;
}

float64_t CStreamingMMD::compute_statistic()
{
	/* use wrapper method and compute for single kernel */
	SGVector<float64_t> statistic;
	SGVector<float64_t> variance;
	compute_statistic_and_variance(statistic, variance, false);

	return statistic[0];
}

SGVector<float64_t> CStreamingMMD::compute_statistic(bool multiple_kernels)
{
	/* make sure multiple_kernels flag is used only with a combined kernel */
	REQUIRE(!multiple_kernels || m_kernel->get_kernel_type()==K_COMBINED,
			"multiple kernels specified, but underlying kernel is not of type "
			"K_COMBINED\n");

	SGVector<float64_t> statistic;
	SGVector<float64_t> variance;
	compute_statistic_and_variance(statistic, variance, multiple_kernels);

	return statistic;
}

float64_t CStreamingMMD::compute_variance_estimate()
{
	/* use wrapper method and compute for single kernel */
	SGVector<float64_t> statistic;
	SGVector<float64_t> variance;
	compute_statistic_and_variance(statistic, variance, false);

	return variance[0];
}

float64_t CStreamingMMD::compute_p_value(float64_t statistic)
{
	float64_t result=0;

	switch (m_null_approximation_method)
	{
	case MMD1_GAUSSIAN:
		{
			/* compute variance and use to estimate Gaussian distribution */
			float64_t std_dev=CMath::sqrt(compute_variance_estimate());
			result=1.0-CStatistics::normal_cdf(statistic, std_dev);
		}
		break;

	default:
		/* sampling null is handled here */
		result=CKernelTwoSampleTest::compute_p_value(statistic);
		break;
	}

	return result;
}

float64_t CStreamingMMD::compute_threshold(float64_t alpha)
{
	float64_t result=0;

	switch (m_null_approximation_method)
	{
	case MMD1_GAUSSIAN:
		{
			/* compute variance and use to estimate Gaussian distribution */
			float64_t std_dev=CMath::sqrt(compute_variance_estimate());
			result=1.0-CStatistics::inverse_normal_cdf(1-alpha, 0, std_dev);
		}
		break;

	default:
		/* sampling null is handled here */
		result=CKernelTwoSampleTest::compute_threshold(alpha);
		break;
	}

	return result;
}

float64_t CStreamingMMD::perform_test()
{
	float64_t result=0;

	switch (m_null_approximation_method)
	{
	case MMD1_GAUSSIAN:
		{
			/* compute variance and use to estimate Gaussian distribution, use
			 * wrapper method and compute for single kernel */
			SGVector<float64_t> statistic;
			SGVector<float64_t> variance;
			compute_statistic_and_variance(statistic, variance, false);

			/* estimate Gaussian distribution */
			result=1.0-CStatistics::normal_cdf(statistic[0],
					CMath::sqrt(variance[0]));
		}
		break;

	default:
		/* sampling null can be done separately in superclass */
		result=CHypothesisTest::perform_test();
		break;
	}

	return result;
}

SGVector<float64_t> CStreamingMMD::sample_null()
{
	SGVector<float64_t> samples(m_num_null_samples);

	/* instead of permutating samples, just samples new data all the time. */
	CStreamingFeatures* p=m_streaming_p;
	CStreamingFeatures* q=m_streaming_q;
	SG_REF(p);
	SG_REF(q);

	bool old=m_simulate_h0;
	set_simulate_h0(true);
	for (index_t i=0; i<m_num_null_samples; ++i)
	{
		/* compute statistic for this permutation of mixed samples */
		samples[i]=compute_statistic();
	}
	set_simulate_h0(old);
	m_streaming_p=p;
	m_streaming_q=q;
	SG_UNREF(p);
	SG_UNREF(q);

	return samples;
}

CList* CStreamingMMD::stream_data_blocks(index_t num_blocks,
		index_t num_this_run)
{
	SG_DEBUG("entering!\n");

	/* the list of blocks of data to be returned, turning delete_data flag
	 * on which SG_REFs the elements when appended or returned. */
	CList* data=new CList(true);

	SG_DEBUG("streaming %d blocks from p of blocksize %d!\n", num_blocks,
			num_this_run);

	/* stream data from p num_blocks of time*/
	for (index_t i=0; i<num_blocks; ++i)
	{
		CFeatures* block=m_streaming_p->get_streamed_features(num_this_run);
		data->append_element(block);
	}

	SG_DEBUG("streaming %d blocks from q of blocksize %d!\n", num_blocks,
			num_this_run);

	/* stream data from q num_blocks of time*/
	for (index_t i=0; i<num_blocks; ++i)
	{
		CFeatures* block=m_streaming_q->get_streamed_features(num_this_run);
		data->append_element(block);
	}

	/* check whether h0 should be simulated and permute if so */
	if (m_simulate_h0)
	{
		/* create merged copy of all feature instances to permute */
		SG_DEBUG("merging and premuting features!\n");

		/* use the first element to merge rest of the data into */
		CFeatures* first=(CFeatures*)data->get_first_element();

		/* this delete element doesn't deallocate first element but just removes
		 * from the list and does a SG_UNREF. But its not deleted because
		 * get_first_element() does a SG_REF before returning so we need to later
		 * manually take care of its destruction via SG_UNREF here itself */
		data->delete_element();

		CFeatures* merged=first->create_merged_copy(data);

		/* now we can get rid of unnecessary feature objects */
		SG_UNREF(first);
		data->delete_all_elements();

		/* permute */
		SGVector<index_t> inds(merged->get_num_vectors());
		inds.range_fill();
		inds.permute();
		merged->add_subset(inds);

		/* copy back */
		SGVector<index_t> copy(num_this_run);
		copy.range_fill();
		for (index_t i=0; i<2*num_blocks; ++i)
		{
			CFeatures* current=merged->copy_subset(copy);
			data->append_element(current);
			/* SG_UNREF'ing since copy_subset does a SG_REF, this is
			 * safe since the object is already SG_REF'ed inside the list */
			SG_UNREF(current);

			if (i<2*num_blocks-1)
				copy.add(num_this_run);
		}

		/* clean up */
		SG_UNREF(merged);
	}

	SG_REF(data);

	SG_DEBUG("leaving!\n");
	return data;
}

void CStreamingMMD::set_p_and_q(CFeatures* p_and_q)
{
	SG_ERROR("Method not implemented since linear time mmd is based on "
			"streaming features\n");
}

CFeatures* CStreamingMMD::get_p_and_q()
{
	SG_ERROR("Method not implemented since linear time mmd is based on "
			"streaming features\n");
	return NULL;
}

CStreamingFeatures* CStreamingMMD::get_streaming_p()
{
	SG_REF(m_streaming_p);
	return m_streaming_p;
}

CStreamingFeatures* CStreamingMMD::get_streaming_q()
{
	SG_REF(m_streaming_q);
	return m_streaming_q;
}

