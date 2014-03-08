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

#include <shogun/statistics/LinearTimeMMD.h>
#include <shogun/features/Features.h>
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/lib/List.h>

#include <shogun/lib/external/libqp.h>

using namespace shogun;

CLinearTimeMMD::CLinearTimeMMD() : CStreamingMMD()
{
}

CLinearTimeMMD::CLinearTimeMMD(CKernel* kernel, CStreamingFeatures* p,
		CStreamingFeatures* q, index_t m, index_t blocksize)
	: CStreamingMMD(kernel, p, q, m, blocksize)
{
}

CLinearTimeMMD::~CLinearTimeMMD()
{
}

void CLinearTimeMMD::compute_squared_mmd(CKernel* kernel, CList* data,
		SGVector<float64_t>& current, SGVector<float64_t>& pp,
		SGVector<float64_t>& qq, SGVector<float64_t>& pq,
		SGVector<float64_t>& qp, index_t num_this_run)
{
	SG_DEBUG("entering!\n");

	REQUIRE(data->get_num_elements()==4, "Wrong number of blocks!\n");

	/* cast is safe the list is passed inside the class
	 * features will be SG_REF'ed once again by these get methods */
	CFeatures* p1=(CFeatures*)data->get_first_element();
	CFeatures* p2=(CFeatures*)data->get_next_element();
	CFeatures* q1=(CFeatures*)data->get_next_element();
	CFeatures* q2=(CFeatures*)data->get_next_element();

	SG_DEBUG("computing MMD values for current kernel!\n");

	/* compute kernel matrix diagonals */
	kernel->init(p1, p2);
	kernel->get_kernel_diagonal(pp);

	kernel->init(q1, q2);
	kernel->get_kernel_diagonal(qq);

	kernel->init(p1, q2);
	kernel->get_kernel_diagonal(pq);

	kernel->init(q1, p2);
	kernel->get_kernel_diagonal(qp);

	/* cleanup */
	SG_UNREF(p1);
	SG_UNREF(p2);
	SG_UNREF(q1);
	SG_UNREF(q2);

	/* compute sum of current h terms for current kernel */

	for (index_t i=0; i<num_this_run; ++i)
		current[i]=pp[i]+qq[i]-pq[i]-qp[i];

	SG_DEBUG("leaving!\n");
}

SGVector<float64_t> CLinearTimeMMD::compute_squared_mmd(CKernel* kernel,
		CList* data, index_t num_this_run)
{
	/* wrapper method used for convenience for using preallocated memory */
	SGVector<float64_t> current(num_this_run);
	SGVector<float64_t> pp(num_this_run);
	SGVector<float64_t> qq(num_this_run);
	SGVector<float64_t> pq(num_this_run);
	SGVector<float64_t> qp(num_this_run);
	compute_squared_mmd(kernel, data, current, pp, qq, pq, qp, num_this_run);
	return current;
}

void CLinearTimeMMD::compute_statistic_and_variance(
		SGVector<float64_t>& statistic, SGVector<float64_t>& variance,
		bool multiple_kernels)
{
	SG_DEBUG("entering!\n")

	REQUIRE(m_streaming_p, "streaming features p required!\n");
	REQUIRE(m_streaming_q, "streaming features q required!\n");

	REQUIRE(m_kernel, "kernel needed!\n");

	/* make sure multiple_kernels flag is used only with a combined kernel */
	REQUIRE(!multiple_kernels || m_kernel->get_kernel_type()==K_COMBINED,
			"multiple kernels specified, but underlying kernel is not of type "
			"K_COMBINED\n");

	/* m is number of samples from each distribution, m_2 is half of it
	 * using names from JLMR paper (see class documentation) */
	index_t m_2=m_m/2;

	SG_DEBUG("m_m=%d\n", m_m)

	/* find out whether single or multiple kernels (cast is safe, check above) */
	index_t num_kernels=1;
	if (multiple_kernels)
	{
		num_kernels=((CCombinedKernel*)m_kernel)->get_num_subkernels();
		SG_DEBUG("computing MMD and variance for %d sub-kernels\n",
				num_kernels);
	}

	/* allocate memory for results if vectors are empty */
	if (!statistic.vector)
		statistic=SGVector<float64_t>(num_kernels);

	if (!variance.vector)
		variance=SGVector<float64_t>(num_kernels);

	/* ensure right dimensions */
	REQUIRE(statistic.vlen==num_kernels,
			"statistic vector size (%d) does not match number of kernels (%d)\n",
			 statistic.vlen, num_kernels);

	REQUIRE(variance.vlen==num_kernels,
			"variance vector size (%d) does not match number of kernels (%d)\n",
			 variance.vlen, num_kernels);

	/* temp variable in the algorithm */
	float64_t delta;

	/* initialise statistic and variance since they are cumulative */
	statistic.zero();
	variance.zero();

	/* needed for online mean and variance */
	SGVector<index_t> term_counters(num_kernels);
	term_counters.set_const(1);

	/* term counter to compute online mean and variance */
	index_t num_examples_processed=0;
	while (num_examples_processed<m_2)
	{
		/* number of example to look at in this iteration */
		index_t num_this_run=CMath::min(m_blocksize,
				CMath::max(0, m_2-num_examples_processed));
		SG_DEBUG("processing %d more examples. %d so far processed. Blocksize "
				"is %d\n", num_this_run, num_examples_processed, m_blocksize);

		/* stream 2 data blocks from each distribution */
		CList* data=stream_data_blocks(2, num_this_run);

		/* if multiple kernels are used, compute all of them on streamed data,
		 * if multiple kernels flag is false, the above loop will be executed
		 * only once */
		CKernel* kernel=m_kernel;
		if (multiple_kernels)
			SG_DEBUG("using multiple kernels\n");

		/* iterate through all kernels for this data */

		for (index_t i=0; i<num_kernels; ++i)
		{
			/* if multiple kernels should be computed, set next kernel */
			if (multiple_kernels)
				kernel=((CCombinedKernel*)m_kernel)->get_kernel(i);

			/* compute linear time MMD values */
			SGVector<float64_t> current=compute_squared_mmd(kernel, data,
					num_this_run);

			/* single variances for all kernels. Update mean and variance
			 * using Knuth's online variance algorithm.
			 * C.f. for example Wikipedia */
			for (index_t j=0; j<num_this_run; ++j)
			{
				/* D. Knuth's online variance algorithm for current kernel */
				delta=current[j]-statistic[i];
				statistic[i]+=delta/term_counters[i]++;
				variance[i]+=delta*(current[j]-statistic[i]);

				SG_DEBUG("burst: current=%f, delta=%f, statistic=%f, "
						"variance=%f, kernel_idx=%d\n", current[j], delta,
						statistic[i], variance[i], i);
			}

			if (multiple_kernels)
				SG_UNREF(kernel);
		}

		/* clean up streamed data, this frees the feature objects  */
		SG_UNREF(data);

		/* add number of processed examples for this run */
		num_examples_processed+=num_this_run;
	}
	SG_DEBUG("Done compouting statistic, processed 2*%d examples.\n",
			num_examples_processed);

	/* mean of sum all traces is linear time mmd, copy entries for all kernels */
	if (io->get_loglevel()==MSG_DEBUG || io->get_loglevel()==MSG_GCDEBUG)
		statistic.display_vector("statistics");

	/* variance of terms can be computed using mean (statistic).
	 * Note that the variance needs to be divided by m_2 in order to get
	 * variance of null-distribution */
	for (index_t i=0; i<num_kernels; ++i)
		variance[i]=variance[i]/(m_2-1)/m_2;

	if (io->get_loglevel()==MSG_DEBUG || io->get_loglevel()==MSG_GCDEBUG)
		variance.display_vector("variances");

	SG_DEBUG("leaving!\n")
}

void CLinearTimeMMD::compute_statistic_and_Q(
		SGVector<float64_t>& statistic, SGMatrix<float64_t>& Q)
{
	SG_DEBUG("entering!\n")

	REQUIRE(m_streaming_p, "streaming features p required!\n");
	REQUIRE(m_streaming_q, "streaming features q required!\n");

	REQUIRE(m_kernel, "kernel needed!\n");

	/* make sure multiple_kernels flag is used only with a combined kernel */
	REQUIRE(m_kernel->get_kernel_type()==K_COMBINED,
			"underlying kernel is not of type K_COMBINED\n");

	/* cast combined kernel */
	CCombinedKernel* combined=(CCombinedKernel*)m_kernel;

	/* m is number of samples from each distribution, m_4 is quarter of it */
	REQUIRE(m_m>=4, "Need at least m>=4\n");
	index_t m_4=m_m/4;

	SG_DEBUG("m_m=%d\n", m_m)

	/* find out whether single or multiple kernels (cast is safe, check above) */
	index_t num_kernels=combined->get_num_subkernels();
	REQUIRE(num_kernels>0, "At least one kernel is needed\n");

	/* allocate memory for results if vectors are empty */
	if (!statistic.vector)
		statistic=SGVector<float64_t>(num_kernels);

	if (!Q.matrix)
		Q=SGMatrix<float64_t>(num_kernels, num_kernels);

	/* ensure right dimensions */
	REQUIRE(statistic.vlen==num_kernels,
			"statistic vector size (%d) does not match number of kernels (%d)\n",
			 statistic.vlen, num_kernels);

	REQUIRE(Q.num_rows==num_kernels,
			"Q number of rows does (%d) not match number of kernels (%d)\n",
			 Q.num_rows, num_kernels);

	REQUIRE(Q.num_cols==num_kernels,
			"Q number of columns (%d) does not match number of kernels (%d)\n",
			 Q.num_cols, num_kernels);

	/* initialise statistic and variance since they are cumulative */
	statistic.zero();
	Q.zero();

	/* produce two kernel lists to iterate doubly nested */
	CList* list_i=new CList();
	CList* list_j=new CList();

	for (index_t k_idx=0; k_idx<combined->get_num_kernels(); k_idx++)
	{
		CKernel* kernel = combined->get_kernel(k_idx);
		list_i->append_element(kernel);
		list_j->append_element(kernel);
		SG_UNREF(kernel);
	}

	/* needed for online mean and variance */
	SGVector<index_t> term_counters_statistic(num_kernels);
	SGMatrix<index_t> term_counters_Q(num_kernels, num_kernels);
	term_counters_statistic.set_const(1);
	term_counters_Q.set_const(1);

	index_t num_examples_processed=0;
	while (num_examples_processed<m_4)
	{
		/* number of example to look at in this iteration */
		index_t num_this_run=CMath::min(m_blocksize,
				CMath::max(0, m_4-num_examples_processed));
		SG_DEBUG("processing %d more examples. %d so far processed. Blocksize "
				"is %d\n", num_this_run, num_examples_processed, m_blocksize);

		/* stream 4 data blocks from each distribution */
		CList* data=stream_data_blocks(4, num_this_run);

		/* create two sets of data, a and b, from alternative blocks */
		CList* data_a=new CList(true);
		CList* data_b=new CList(true);

		/* take care of refcounts */
		int32_t num_elements=data->get_num_elements();
		CFeatures* current=(CFeatures*)data->get_first_element();
		data_a->append_element(current);
		SG_UNREF(current);
		current=(CFeatures*)data->get_next_element();
		data_b->append_element(current);
		SG_UNREF(current);
		num_elements-=2;
		/* loop counter is safe since num_elements can only be even */
		while (num_elements)
		{
			current=(CFeatures*)data->get_next_element();
			data_a->append_element(current);
			SG_UNREF(current);
			current=(CFeatures*)data->get_next_element();
			data_b->append_element(current);
			SG_UNREF(current);
			num_elements-=2;
		}
		/* safely unref previous list of data, decreases refcounts of features
		 * but doesn't delete them */
		SG_UNREF(data);

		/* now for each of these streamed data instances, iterate through all
		 * kernels and update Q matrix while also computing MMD statistic */

		/* preallocate some memory for faster processing */
		SGVector<float64_t> pp(num_this_run);
		SGVector<float64_t> qq(num_this_run);
		SGVector<float64_t> pq(num_this_run);
		SGVector<float64_t> qp(num_this_run);
		SGVector<float64_t> h_i_a(num_this_run);
		SGVector<float64_t> h_i_b(num_this_run);
		SGVector<float64_t> h_j_a(num_this_run);
		SGVector<float64_t> h_j_b(num_this_run);

		/* iterate through Q matrix and update values, compute mmd */
		CKernel* kernel_i=(CKernel*)list_i->get_first_element();
		for (index_t i=0; i<num_kernels; ++i)
		{
			/* compute all necessary 8 h-vectors for this burst.
			 * h_delta-terms for each kernel, expression 7 of NIPS paper */

			/* first kernel, a-part */
			compute_squared_mmd(kernel_i, data_a, h_i_a, pp, qq, pq, qp,
					num_this_run);

			/* first kernel, b-part */
			compute_squared_mmd(kernel_i, data_b, h_i_b, pp, qq, pq, qp,
					num_this_run);

			/* iterate through j, but use symmetry in order to save half of the
			 * computations */
			CKernel* kernel_j=(CKernel*)list_j->get_first_element();
			for (index_t j=0; j<=i; ++j)
			{
				/* compute all necessary 8 h-vectors for this burst.
				 * h_delta-terms for each kernel, expression 7 of NIPS paper */

				/* second kernel, a-part */
				compute_squared_mmd(kernel_j, data_a, h_j_a, pp, qq, pq, qp,
						num_this_run);

				/* second kernel, b-part */
				compute_squared_mmd(kernel_j, data_b, h_j_b, pp, qq, pq, qp,
						num_this_run);

				float64_t term;
				for (index_t it=0; it<num_this_run; ++it)
				{
					/* current term of expression 7 of NIPS paper */
					term=(h_i_a[it]-h_i_b[it])*(h_j_a[it]-h_j_b[it]);

					/* update covariance element for the current burst. This is a
					 * running average of the product of the h_delta terms of each
					 * kernel */
					Q(i, j)+=(term-Q(i, j))/term_counters_Q(i, j)++;
				}

				/* use symmetry */
				Q(j, i)=Q(i, j);

				/* next kernel j */
				kernel_j=(CKernel*)list_j->get_next_element();
			}

			/* update MMD statistic online computation for kernel i, using
			 * vectors that were computed above */
			SGVector<float64_t> h(num_this_run*2);
			for (index_t it=0; it<num_this_run; ++it)
			{
				/* update statistic for kernel i (outer loop) and update using
				 * all elements of the h_i_a, h_i_b vectors (iterate over it) */
				statistic[i]=statistic[i]+
						(h_i_a[it]-statistic[i])/term_counters_statistic[i]++;

				/* Make sure to use all data, i.e. part a and b */
				statistic[i]=statistic[i]+
						(h_i_b[it]-statistic[i])/(term_counters_statistic[i]++);
			}

			/* next kernel i */
			kernel_i=(CKernel*)list_i->get_next_element();
		}

		/* clean up streamed data */
		SG_UNREF(data_a);
		SG_UNREF(data_b);

		/* add number of processed examples for this run */
		num_examples_processed+=num_this_run;
	}

	/* clean up */
	SG_UNREF(list_i);
	SG_UNREF(list_j);

	SG_DEBUG("Done compouting statistic, processed 4*%d examples.\n",
			num_examples_processed);

	SG_DEBUG("leaving!\n")
}

