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

#include <shogun/statistics/TwoSampleTest.h>
#include <shogun/features/Features.h>

using namespace shogun;

CTwoSampleTest::CTwoSampleTest() : CHypothesisTest()
{
	init();
}

CTwoSampleTest::CTwoSampleTest(CFeatures* p_and_q, index_t m) :
	CHypothesisTest()
{
	init();

	m_p_and_q=p_and_q;
	SG_REF(m_p_and_q);

	m_m=m;
}

CTwoSampleTest::CTwoSampleTest(CFeatures* p, CFeatures* q) :
	CHypothesisTest()
{
	init();

	m_p_and_q=p->create_merged_copy(q);
	SG_REF(m_p_and_q);

	m_m=p->get_num_vectors();
}

CTwoSampleTest::~CTwoSampleTest()
{
	SG_UNREF(m_p_and_q);
}

void CTwoSampleTest::init()
{
	SG_ADD((CSGObject**)&m_p_and_q, "p_and_q", "Concatenated samples p and q",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_m, "m", "Index of first sample of q",
			MS_NOT_AVAILABLE);

	m_p_and_q=NULL;
	m_m=0;
}

SGVector<float64_t> CTwoSampleTest::sample_null()
{
	SG_DEBUG("entering!\n")

	REQUIRE(m_p_and_q, "No appended features p and q!\n");

	/* compute sample statistics for null distribution */
	SGVector<float64_t> results(m_num_null_samples);

	/* memory for index permutations. Adding of subset has to happen
	 * inside the loop since it may be copied if there already is one set */
	SGVector<index_t> ind_permutation(m_p_and_q->get_num_vectors());
	ind_permutation.range_fill();

	for (index_t i=0; i<m_num_null_samples; ++i)
	{
		/* idea: merge features of p and q, shuffle, and compute statistic.
		 * This is done using subsets here */

		/* create index permutation and add as subset. This will mix samples
		 * from p and q */
		SGVector<index_t>::permute_vector(ind_permutation);

		/* compute statistic for this permutation of mixed samples */
		m_p_and_q->add_subset(ind_permutation);
		results[i]=compute_statistic();
		m_p_and_q->remove_subset();
	}

	SG_DEBUG("leaving!\n")
	return results;
}

float64_t CTwoSampleTest::compute_p_value(float64_t statistic)
{
	float64_t result=0;

	if (m_null_approximation_method==PERMUTATION)
	{
		/* sample a bunch of MMD values from null distribution */
		SGVector<float64_t> values=sample_null();

		/* find out percentile of parameter "statistic" in null distribution */
		values.qsort();
		float64_t i=values.find_position_to_insert(statistic);

		/* return corresponding p-value */
		result=1.0-i/values.vlen;
	}
	else
		SG_ERROR("Unknown method to approximate null distribution!\n");

	return result;
}

float64_t CTwoSampleTest::compute_threshold(float64_t alpha)
{
	float64_t result=0;

	if (m_null_approximation_method==PERMUTATION)
	{
		/* sample a bunch of MMD values from null distribution */
		SGVector<float64_t> values=sample_null();

		/* return value of (1-alpha) quantile */
		result=values[index_t(CMath::floor(values.vlen*(1-alpha)))];
	}
	else
		SG_ERROR("Unknown method to approximate null distribution!\n");

	return result;
}

void CTwoSampleTest::set_p_and_q(CFeatures* p_and_q)
{
	/* ref before unref to avoid problems when instances are equal */
	SG_REF(p_and_q);
	SG_UNREF(m_p_and_q);
	m_p_and_q=p_and_q;
}

void CTwoSampleTest::set_m(index_t m)
{
	REQUIRE(m_p_and_q, "Samples are not specified!\n");
	REQUIRE(m_p_and_q->get_num_vectors()>m, "Provided sample size for p"
			"(%d) is greater than total number of samples (%d)!\n",
			m, m_p_and_q->get_num_vectors());
	m_m=m;
}

CFeatures* CTwoSampleTest::get_p_and_q()
{
	SG_REF(m_p_and_q);
	return m_p_and_q;
}

