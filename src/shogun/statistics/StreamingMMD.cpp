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
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/lib/List.h>

using namespace shogun;

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;
#endif // HAVE_EIGEN3

CStreamingMMD::CStreamingMMD() : CKernelTwoSampleTest()
{
	init();
}

CStreamingMMD::CStreamingMMD(CKernel* kernel, CStreamingFeatures* p,
		CStreamingFeatures* q, index_t m, index_t n)
: CKernelTwoSampleTest(kernel, NULL, m)
{
	init();

	m_streaming_p=p;
	SG_REF(m_streaming_p);

	m_streaming_q=q;
	SG_REF(m_streaming_q);

	m_n=n;
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
	SG_ADD(&m_blocksize_p, "blocksize_p", "Number of samples from p processed "
			"at once", MS_NOT_AVAILABLE);
	SG_ADD(&m_blocksize_q, "blocksize_q", "Number of samples from q processed "
			"at once", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_blocks, "num_blocks", "Number of blocks streamed in one burst",
				MS_NOT_AVAILABLE);
	SG_ADD(&m_n, "n", "Number of samples from second distribution",
				MS_NOT_AVAILABLE);
	SG_ADD(&m_simulate_h0, "simulate_h0", "Whether p and q are mixed",
				MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*)&m_statistic_type, "statistic_type",
			"Statistic estimation type for streaming MMD", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*)&m_null_var_est_method, "null_var_est_method",
			"Estimation method for variance under null", MS_NOT_AVAILABLE);

	m_streaming_p=NULL;
	m_streaming_q=NULL;
	m_blocksize=0;
	m_blocksize_p=0;
	m_blocksize_q=0;
	m_num_blocks=1000;
	m_n=0;
	m_simulate_h0=false;
	m_statistic_type=S_UNBIASED;
	m_null_var_est_method=WITHIN_BLOCK_PERMUTATION;
}

SGVector<float64_t> CStreamingMMD::compute_blockwise_statistic(CKernel* kernel,
		CFeatures* p_and_q_current_burst)
{
	SG_DEBUG("Entering!\n")

	/* results, one entry per block */
	SGVector<float64_t> mmds(m_num_blocks);

	index_t Bx=m_blocksize_p;
	index_t By=m_blocksize_q;

	/* init kernel with features for all blocks in the current burst.
	   NULL check is handled in compute_statistic */
	kernel->init(p_and_q_current_burst, p_and_q_current_burst);

	index_t offset=Bx*m_num_blocks;

	/* compute blockwise estimate for each m_num_blocks */
	for (index_t i=0; i<m_num_blocks; ++i)
	{
		SG_DEBUG("Processing block %d of %d of the current burst!\n",
				i+1, m_num_blocks);

		/* compute kernel values for the next block and their sum on the go */
		float64_t xx_sum=kernel->sum_symmetric_block(Bx*i, Bx);
		float64_t yy_sum=kernel->sum_symmetric_block(offset+By*i, By);

		float64_t xy_sum=0.0;

		/* remove diagonal entries if statistic type is incomplete */
		if (m_statistic_type==S_UNBIASED)
			xy_sum=kernel->sum_block(offset+By*i, Bx*i, By, Bx);
		else if (m_statistic_type==S_INCOMPLETE)
			xy_sum=kernel->sum_block(offset+By*i, Bx*i, By, Bx, true);
		else
			SG_ERROR("Unknown statistic type %d!\n", m_statistic_type);

		/* split computations into three terms from JLMR paper (see documentation )*/

		/* first term */
		float64_t first=xx_sum/Bx/(Bx-1);

		/* second term */
		float64_t second=yy_sum/By/(By-1);

		/* third term */
		float64_t third=2.0*xy_sum;
		if (m_statistic_type==S_UNBIASED)
			third/=Bx*By;
		else
			third/=Bx*(Bx-1);

		/* finally computing the statistic */
		mmds[i]=first+second-third;

		SG_INFO("Computed statistic for current block!\n");
		SG_DEBUG("statistic(%f)=first(%f)+second(%f)-third(%f)\n", mmds[i],
				first, second, third);
	}

	/* release feature objects */
	kernel->remove_lhs_and_rhs();

	SG_DEBUG("Leaving!\n")

	return mmds;
}

// TODO replace this with linalg methods when its merged with develop
#ifdef HAVE_EIGEN3
SGMatrix<float64_t> CStreamingMMD::compute_blockwise_statistic_variance(CKernel*
		kernel, CFeatures* p_and_q_current_burst)
{
	SG_DEBUG("Entering!\n")

	/* results, one row per block. first entry statistic estimate, 2nd entry
	   variance estimate */
	SGMatrix<float64_t> results(m_num_blocks, 2);

	index_t B=m_blocksize;
	index_t Bx=m_blocksize_p;
	index_t By=m_blocksize_q;

	/* since the feature provided is merged samples of m_num_blocks blocks
	   from both distributions, we use subset to process this per block to
	   compute mmd estimate */
	SGVector<index_t> inds(m_blocksize);
	for (index_t i=0; i<Bx; ++i)
		inds[i]=i;
	for (index_t i=0; i<By; ++i)
		inds[i+Bx]=i+Bx*m_num_blocks;

	/* compute blockwise estimate for each m_num_blocks */
	for (index_t i=0; i<m_num_blocks; ++i)
	{
		SG_DEBUG("Processing block %d of %d of the current burst! Samples being"
				" processed are\n", i+1, m_num_blocks);
		if (io->get_loglevel()==MSG_DEBUG || io->get_loglevel()==MSG_GCDEBUG)
			inds.display_vector("feat indices");

		p_and_q_current_burst->add_subset(inds);

		/* init kernel with features. NULL check is handled in compute_statistic */
		kernel->init(p_and_q_current_burst, p_and_q_current_burst);

		/* get the kernel matrix, required for computing matrix-matrix product */
		SGMatrix<float64_t> km=kernel->get_kernel_matrix();

		/* release feature objects */
		kernel->remove_lhs_and_rhs();

		Map<MatrixXd> k_m(km.matrix, km.num_rows, km.num_cols);
		k_m.diagonal().setZero();

		/* compute kernel values and their sum on the go */
		float64_t xx_sum=k_m.block(0, 0, Bx, Bx).sum();
		float64_t yy_sum=k_m.block(Bx, Bx, By, By).sum();

		float64_t xy_sum=k_m.block(0, Bx, Bx, By).sum();

		/* remove diagonal entries if statistic type is incomplete */
		if (m_statistic_type==S_INCOMPLETE)
			xy_sum-=k_m.block(0, Bx, Bx, By).diagonal().sum();

		/* computing statistic estimate */

		/* split computations into three terms from JLMR paper (see documentation )*/

		/* first term */
		float64_t first=xx_sum/Bx/(Bx-1);

		/* second term */
		float64_t second=yy_sum/By/(By-1);

		/* third term */
		float64_t third=2.0*xy_sum;
		if (m_statistic_type==S_UNBIASED)
			third/=Bx*By;
		else
			third/=Bx*(Bx-1);

		/* finally computing the statistic */
		results(i,0)=first+second-third;

		SG_INFO("Computed statistic for current block!\n");
		SG_DEBUG("statistic(%f)=first(%f)+second(%f)-third(%f)\n", results(i,0),
			first, second, third);

		/* computing variance estimate */

		/* split computations into three terms (see documentation )*/

		/* first term */
		first=k_m.array().square().sum();

		/* second term */
		second=CMath::sq(k_m.sum())/(B-1)/(B-2);

		/* third term */
		third=2*(k_m*k_m).sum()/(B-2);

		/* finally computing the variance */
		results(i,1)=2.0/B/(B-3)*(first+second-third);

		SG_INFO("Computed variance for current block!\n");
		SG_DEBUG("first=%f, second=%f, third=%f, variance=%f\n", first, second,
			third, results(i,1));

		/* prepare for next block */
		p_and_q_current_burst->remove_subset();
		for (index_t j=0; j<Bx; ++j)
			inds[j]+=Bx;
		for (index_t j=0; j<By; ++j)
			inds[j+Bx]+=By;
	}

	SG_DEBUG("Leaving!\n")

	return results;
}
#endif // HAVE_EIGEN3

void CStreamingMMD::compute_statistic_and_variance(SGVector<float64_t>&
		statistic, SGVector<float64_t>& variance, bool multiple_kernels)
{
	SG_DEBUG("Entering!\n")

	REQUIRE(m_streaming_p, "Streaming features p required!\n");
	REQUIRE(m_streaming_q, "Streaming features q required!\n");

	REQUIRE(m_kernel, "Kernel not initialized!\n");

	/* make sure multiple_kernels flag is used only with a combined kernel */
	REQUIRE(!multiple_kernels || m_kernel->get_kernel_type()==K_COMBINED,
			"Multiple kernels specified, but underlying kernel is not of type "
			"K_COMBINED\n");

	/* find out whether single or multiple kernels (cast is safe, check above) */
	index_t num_kernels=1;
	if (multiple_kernels)
	{
		num_kernels=((CCombinedKernel*)m_kernel)->get_num_subkernels();
		SG_DEBUG("Computing MMD and variance for %d sub-kernels\n",
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

	/* temp variable in the algorithm for estimating variance under null */
	SGVector<float64_t> temp(num_kernels);

	/* initialise statistic, variance and temp since they are cumulative */
	statistic.zero();
	variance.zero();
	temp.zero();

	/* needed for online mean and variance */
	SGVector<index_t> term_counters(num_kernels);
	term_counters.set_const(1);

	index_t total_num_examples=m_m+m_n;
	SG_DEBUG("Total %d examples are to be processed!\n", total_num_examples);

	/* term counter to compute online mean and variance */
	index_t num_examples_processed=0;
	while (num_examples_processed<total_num_examples)
	{
		/* number of blocks to stream at in this burst */
		index_t num_blocks_this_run=CMath::min(m_num_blocks,(total_num_examples
				-num_examples_processed)/m_blocksize);

		/* temporarily replace num_blocks */
		index_t orig_num_blocks=m_num_blocks;
		m_num_blocks=num_blocks_this_run;

		SG_DEBUG("Processing %d more examples. %d so far processed!\n",
				m_blocksize*m_num_blocks, num_examples_processed);

		/* stream blocks from each distribution. data is merged samples */
		CFeatures* data=stream_data_blocks();

		/* if multiple kernels are used, compute all of them on streamed data */
		CKernel* kernel=m_kernel;
		if (multiple_kernels)
			SG_DEBUG("Using multiple kernels\n");

		/* iterate through all kernels for this data. if multiple kernels flag
		 * is false, the following loop will be executed only once */
		for (index_t i=0; i<num_kernels; ++i)
		{
			/* if multiple kernels should be computed, set next kernel */
			if (multiple_kernels)
				kernel=((CCombinedKernel*)m_kernel)->get_kernel(i);

			if (m_null_var_est_method==WITHIN_BLOCK_PERMUTATION)
			{
				/* compute blockwise statistic */
				SGVector<float64_t> current
					=compute_blockwise_statistic(kernel, data);

				SG_DEBUG("Permuting the samples for estimating the variance "
						"under null using within-block permutation method!\n")

				/* randomly permute the samples using index permutation. This
				 * is equivalent to splitting the data randomly in the same
				 * proportions between p and q for current block */
				SGVector<index_t> inds(data->get_num_vectors());
				inds.range_fill();
				inds.permute();
				data->add_subset(inds);
				SGVector<float64_t> shuffled
					=compute_blockwise_statistic(kernel, data);
				data->remove_subset();

				/* single variances for all kernels. Update mean and variance
				 * using Knuth's online variance algorithm.
				 * C.f. for example Wikipedia */

				for (index_t j=0; j<m_num_blocks; ++j)
				{
					/* compute online mean of blockwise statistic estimates */
					float64_t delta=current[j]-statistic[i];
					statistic[i]+=delta/term_counters[i];
					SG_DEBUG("Burst: current=%f, delta=%f, statistic=%f\n", current[j],
							delta, statistic[i]);

					/* D. Knuth's online variance algorithm for current kernel */
					delta=shuffled[j]-temp[i];
					temp[i]+=delta/term_counters[i];
					variance[i]+=delta*(shuffled[j]-temp[i]);
					SG_DEBUG("Burst: shuffled=%f, delta=%f, mean=%f, variance %f\n",
							shuffled[j], delta, temp[i], variance[i]);

					/* remember to increament term counters per kernel */
					term_counters[i]++;
				}
			}
			else if(m_null_var_est_method==WITHIN_BLOCK_DIRECT)
			{
				/* compute blockwise statistic and variance */
				SGMatrix<float64_t> current=compute_blockwise_statistic_variance(
						kernel, data);

				SG_DEBUG("Computed statistic and variance under null using "
						"within-block direct estimation method!\n")

				for (index_t j=0; j<m_num_blocks; ++j)
				{
					/* compute online mean of blockwise statistic estimates */
					float64_t delta=current(j,0)-statistic[i];
					statistic[i]+=delta/term_counters[i];
					SG_DEBUG("Burst: current=%f, delta=%f, statistic=%f\n", current(j,0),
							delta, statistic[i]);

					/* compute online mean of blockwise variance estimates */
					delta=current(j,1)-variance[i];
					variance[i]+=delta/term_counters[i];
					SG_DEBUG("Burst: current=%f, delta=%f, variance=%f\n", current(j,1),
							delta, variance[i]);

					/* remember to increament term counters per kernel */
					term_counters[i]++;
				}
			}
			else
				SG_ERROR("Unknown variance estimation method specified\n");

			/* reduces refcounting that was increased by get_kernel call */
			if (multiple_kernels)
				SG_UNREF(kernel);
		}

		/* clean up streamed data, this frees the feature objects  */
		SG_UNREF(data);

		/* add number of processed examples for this run */
		num_examples_processed+=m_blocksize*m_num_blocks;

		/* restore original num_blocks */
		m_num_blocks=orig_num_blocks;
	}

	SG_DEBUG("Done computing statistic, processed %d examples.\n",
			num_examples_processed);

	/* scale the statistic for computing p-value. The multiplier is different
	   for different subclasses */
	statistic.scale(compute_stat_est_multiplier());

	if (io->get_loglevel()==MSG_DEBUG || io->get_loglevel()==MSG_GCDEBUG)
		statistic.display_vector("statistics");

	/* Note that if within-block permutation method is used for estimating
	   the variance, then it needs to be divided by #terms-1 in order to
	   get variance of null-distribution. Note that the multiplier is different
	   for subclasses */
	if (m_null_var_est_method==WITHIN_BLOCK_PERMUTATION)
	{
		float64_t multiplier=compute_var_est_multiplier();
		variance.scale(multiplier/(term_counters[0]-1));
	}

	if (io->get_loglevel()==MSG_DEBUG || io->get_loglevel()==MSG_GCDEBUG)
		variance.display_vector("variances");

	SG_DEBUG("Leaving!\n")
}

void CStreamingMMD::compute_statistic_and_Q(SGVector<float64_t>& statistic,
		SGMatrix<float64_t>& Q)
{
	SG_DEBUG("Entering!\n")

	REQUIRE(m_streaming_p, "Streaming features p required!\n");
	REQUIRE(m_streaming_q, "Streaming features q required!\n");

	if (((m_m+m_n)/m_blocksize)%2!=0)
	{
		SG_ERROR("Only possible when there are even number of blocks!\n"
				"(%d blocks of blocksize %d for %d total samples!)\n",
				(m_m+m_n)/m_blocksize, m_blocksize, m_m+m_n);
	}

	if (m_blocksize_p%2!=0 || m_blocksize_q%2!=0)
	{
		SG_ERROR("Only possible when number of samples from both the "
				"distributions (%d from p and %d from q) within a "
				"block are even!\n", m_blocksize_p, m_blocksize_q);
	}

	REQUIRE(m_kernel, "Kernel not initialized!\n");

	/* make sure multiple_kernels flag is used only with a combined kernel */
	REQUIRE(m_kernel->get_kernel_type()==K_COMBINED,
			"Underlying kernel is not of type K_COMBINED\n");

	/* cast combined kernel */
	CCombinedKernel* combined=(CCombinedKernel*)m_kernel;

	/* find out whether single or multiple kernels */
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

	/* produce two kernel lists to iterate through a doubly nested loop */
	CList* list_i=new CList();
	CList* list_j=new CList();

	for (index_t k_idx=0; k_idx<combined->get_num_kernels(); k_idx++)
	{
		CKernel* kernel=combined->get_kernel(k_idx);
		list_i->append_element(kernel);
		list_j->append_element(kernel);
		SG_UNREF(kernel);
	}

	/* needed for online mean and variance */
	SGVector<index_t> term_counters_statistic(num_kernels);
	SGMatrix<index_t> term_counters_Q(num_kernels, num_kernels);
	term_counters_statistic.set_const(1);
	term_counters_Q.set_const(1);

	index_t total_num_examples=m_m+m_n;
	SG_DEBUG("Total %d examples are to be processed!\n", total_num_examples);

	index_t num_examples_processed=0;

	while (num_examples_processed<total_num_examples)
	{
		/* number of blocks to stream at in this burst */
		index_t num_blocks_this_run=CMath::min(m_num_blocks,(total_num_examples
				-num_examples_processed)/2/m_blocksize);

		/* temporarily replace num_blocks */
		index_t orig_num_blocks=m_num_blocks;
		m_num_blocks=num_blocks_this_run;

		SG_DEBUG("Processing %d more examples. %d so far processed!\n",
				m_blocksize*m_num_blocks, num_examples_processed);

		/* stream 2*m_num_blocks data blocks from each distribution */
		CFeatures* data_a=stream_data_blocks();
		CFeatures* data_b=stream_data_blocks();

		/* required for index permutation for computing covariance */
		SGVector<index_t> inds_a(data_a->get_num_vectors());
		SGVector<index_t> inds_b(data_b->get_num_vectors());

		/* now for each of these streamed data instances, iterate through all
		 * kernels and update Q matrix while also computing MMD statistic */

		/* iterate through Q matrix and update values, compute mmd */
		CKernel* kernel_i=(CKernel*)list_i->get_first_element();
		for (index_t i=0; i<num_kernels; ++i)
		{
			/* compute the statistic on both the a and b-parts */
			SGVector<float64_t> h_i_a
				=compute_blockwise_statistic(kernel_i, data_a);
			SGVector<float64_t> h_i_b
				=compute_blockwise_statistic(kernel_i, data_b);

			/* update MMD statistic online computation for kernel i, using
			 * blockwise estimates that were computed above */

			for (index_t j=0; j<m_num_blocks; ++j)
			{
				/* update statistic for kernel i (outer loop) and update using
			 	* all the h_i_a, h_i_b estimates */
				statistic[i]+=(h_i_a[j]-statistic[i])/term_counters_statistic[i]++;

				/* Make sure to use all data, i.e. part a and b */
				statistic[i]+=(h_i_b[j]-statistic[i])/term_counters_statistic[i]++;
			}

			SG_DEBUG("Permuting the samples for estimating covariance "
					"using within-block permutation method!\n")

			/* randomly permute the samples using index permutation. This
			 * is equivalent to splitting the data randomly in the same
			 * proportions between p and q for current blocks */
			inds_a.range_fill();
			inds_a.permute();
			data_a->add_subset(inds_a);

			inds_b.range_fill();
			inds_b.permute();
			data_b->add_subset(inds_b);

			/* first kernel, a-part */
			h_i_a=compute_blockwise_statistic(kernel_i, data_a);

			/* first kernel, b-part */
			h_i_b=compute_blockwise_statistic(kernel_i, data_b);

			/* iterate through j, but use symmetry in order to save half of the
			 * computations */
			CKernel* kernel_j=(CKernel*)list_j->get_first_element();
			for (index_t j=0; j<=i; ++j)
			{
				/* second kernel, a-part */
				SGVector<float64_t> h_j_a
					=compute_blockwise_statistic(kernel_j, data_a);

				/* second kernel, b-part */
				SGVector<float64_t> h_j_b
					=compute_blockwise_statistic(kernel_j, data_b);

				for (index_t it=0; it<m_num_blocks; ++it)
				{
					float64_t term=(h_i_a[it]-h_i_b[it])*(h_j_a[it]-h_j_b[it]);

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

			/* restore original order of samples for next iteration */
			data_a->remove_subset();
			data_b->remove_subset();

			/* next kernel i */
			kernel_i=(CKernel*)list_i->get_next_element();
		}

		/* clean up streamed data */
		SG_UNREF(data_a);
		SG_UNREF(data_b);

		/* add number of processed examples for this run */
		num_examples_processed+=2*m_blocksize*m_num_blocks;

		/* restore original num_blocks */
		m_num_blocks=orig_num_blocks;
	}

	/* scale the statistic for computing p-value */
	statistic.scale(compute_stat_est_multiplier());

	if (io->get_loglevel()==MSG_DEBUG || io->get_loglevel()==MSG_GCDEBUG)
	{
		statistic.display_vector("statistics");
		Q.display_matrix("Q");
	}

	/* clean up */
	SG_UNREF(list_i);
	SG_UNREF(list_j);

	SG_DEBUG("Done compouting statistic, processed %d examples.\n",
			num_examples_processed);

	SG_DEBUG("Leaving!\n")
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
			"Multiple kernels specified, but underlying kernel is not of type "
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
			float64_t m=compute_gaussian_multiplier();
			float64_t std_dev=CMath::sqrt(m*compute_variance_estimate());
			SG_DEBUG("std_dev = %f\n", std_dev);
			result=1.0-CStatistics::normal_cdf(statistic, std_dev);
		}
		break;

	default:
		/* permutation test is handled here */
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
			float64_t m=compute_gaussian_multiplier();
			float64_t std_dev=CMath::sqrt(m*compute_variance_estimate());
			result=1.0-CStatistics::inverse_normal_cdf(1-alpha, 0, std_dev);
		}
		break;

	default:
		/* permutation test is handled here */
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
			float64_t m=compute_gaussian_multiplier();
			result=1.0-CStatistics::normal_cdf(statistic[0],
					CMath::sqrt(m*variance[0]));
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
	SG_DEBUG("Entering!\n");

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

	SG_DEBUG("Leaving!\n");

	return samples;
}

CFeatures* CStreamingMMD::stream_data_blocks()
{
	SG_DEBUG("Entering!\n");

	/* sanity checks are not required since this method is not availble in the
	 * public API and will only be called from within the class */
	SG_DEBUG("Streaming %dx%d samples from p and %dx%d samples from q!\n",
			m_num_blocks, m_blocksize_p, m_num_blocks, m_blocksize_q);

	/* stream data from p and q */
	CFeatures* first=m_streaming_p->
		get_streamed_features(m_blocksize_p*m_num_blocks);

	CFeatures* second=m_streaming_q->
		get_streamed_features(m_blocksize_q*m_num_blocks);

	CFeatures* merged=first->create_merged_copy(second);

	/* now we can get rid of unnecessary feature objects */
	SG_UNREF(first);
	SG_UNREF(second);

	/* check whether h0 should be simulated and permute if so */
	if (m_simulate_h0)
	{
		SG_DEBUG("Shuffling features for permutation test!\n");

		/* permute */
		SGVector<index_t> inds(merged->get_num_vectors());
		inds.range_fill();
		inds.permute();
		merged->add_subset(inds);
	}

	SG_REF(merged);

	SG_DEBUG("Leaving!\n");

	return merged;
}

void CStreamingMMD::set_p_and_q(CFeatures* p_and_q)
{
	SG_WARNING("Method not implemented since streaming mmd is based on "
			"streaming features\n");
}

CFeatures* CStreamingMMD::get_p_and_q()
{
	SG_WARNING("Method not implemented since streaming mmd is based on "
			"streaming features, returning NULL!\n");
	return NULL;
}

void CStreamingMMD::set_streaming_p(CStreamingFeatures* p)
{
	/* ref before unref to avoid accidental deletion when instances are same */
	SG_REF(p);
	SG_UNREF(m_streaming_p);
	m_streaming_p=p;
}

void CStreamingMMD::set_streaming_q(CStreamingFeatures* q)
{
	/* ref before unref to avoid accidental deletion when instances are same */
	SG_REF(q);
	SG_UNREF(m_streaming_q);
	m_streaming_q=q;
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

void CStreamingMMD::set_blocksize(index_t blocksize)
{
	index_t n=m_m+m_n;

	REQUIRE(blocksize>0 && blocksize<=n, "Blocksize (%d provided) must be "
			"positive and smaller than total number of samples (%d)!\n",
			blocksize, n);

	m_blocksize=blocksize;
	SG_DEBUG("Blocksize set as %d!\n", m_blocksize);

	if(n%m_blocksize!=0)
	{
		SG_ERROR("Total number of samples (%d) is not divisible "
			"by the blocksize (%d)!\n", n, m_blocksize);
	}

	REQUIRE((m_blocksize*m_m)%n==0, "number of samples from p within a block "
		   "is not an integer!\n");
	REQUIRE((m_blocksize*m_n)%n==0, "number of samples from q within a block "
		   "is not an integer!\n");

	m_blocksize_p=m_blocksize*m_m/n;
	m_blocksize_q=m_blocksize*m_n/n;

	// ensure that m_num_blocks is set to a valid number of blocks
	set_num_blocks(m_num_blocks);
}

void CStreamingMMD::set_num_blocks(index_t num_blocks)
{
	REQUIRE(m_blocksize>0, "Blocksize is not set!\n");
	REQUIRE(num_blocks>0, "Number of blocks is not positive (%d)!\n",
			num_blocks);

	index_t max_num_blocks=m_m/m_blocksize_p;

	if (num_blocks>max_num_blocks)
	{
		SG_DEBUG("Provided number of blocks (%d) exceeds the maximum number of "
				"blocks (%d)!\n", num_blocks, max_num_blocks);
		m_num_blocks=max_num_blocks;
	}
	else
		m_num_blocks=num_blocks;

	SG_DEBUG("Number of blocks to be streamed in one burst is set as %d!\n",
			m_num_blocks);
}

index_t CStreamingMMD::get_blocksize()
{
	return m_blocksize;
}

index_t CStreamingMMD::get_num_blocks()
{
	return m_num_blocks;
}

void CStreamingMMD::set_statistic_type(EStreamingStatisticType statistic_type)
{
	m_statistic_type=statistic_type;
	if (m_statistic_type==S_INCOMPLETE)
	{
		REQUIRE(m_m==m_n, "Number of samples should be same with incomplete "
				"statistic estimation!\n");
	}
}

EStreamingStatisticType CStreamingMMD::get_statistic_type()
{
	return m_statistic_type;
}

void CStreamingMMD::set_null_var_est_method(ENullVarianceEstimationMethod
		null_var_est_method)
{
	m_null_var_est_method=null_var_est_method;
	if (m_null_var_est_method==WITHIN_BLOCK_DIRECT)
	{
#ifndef HAVE_EIGEN3
		SG_WARNING("Only possible with Eigen3 installed!\n")
#endif
	}
}

ENullVarianceEstimationMethod CStreamingMMD::get_null_var_est_method()
{
	return m_null_var_est_method;
}

