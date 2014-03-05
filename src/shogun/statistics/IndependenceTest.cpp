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

#include <shogun/statistics/IndependenceTest.h>
#include <shogun/features/Features.h>

using namespace shogun;

CIndependenceTest::CIndependenceTest() : CHypothesisTest()
{
	init();
}

CIndependenceTest::CIndependenceTest(CFeatures* p, CFeatures* q)
	: CHypothesisTest()
{
	init();

	SG_REF(p);
	SG_REF(q);

	m_p=p;
	m_q=q;
}

CIndependenceTest::~CIndependenceTest()
{
	SG_UNREF(m_p);
	SG_UNREF(m_q);
}

void CIndependenceTest::init()
{
	SG_ADD((CSGObject**)&m_p, "p", "Samples from p", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_q, "q", "Samples from q", MS_NOT_AVAILABLE);

	m_p=NULL;
	m_q=NULL;
}

SGVector<float64_t> CIndependenceTest::sample_null()
{
	SG_DEBUG("entering!\n")

	REQUIRE(m_p, "No features p!\n");
	REQUIRE(m_q, "No features q!\n");

	/* compute sample statistics for null distribution */
	SGVector<float64_t> results(m_num_null_samples);

	/* memory for index permutations. Adding of subset has to happen
	 * inside the loop since it may be copied if there already is one set.
	 *
	 * subset for selecting samples from p. In this case we want to
	 * shuffle only samples from p while keeping samples from q fixed */
	SGVector<index_t> ind_permutation(m_p->get_num_vectors());
	ind_permutation.range_fill();

	for (index_t i=0; i<m_num_null_samples; ++i)
	{
		/* idea: shuffle samples from p while keeping samples from q fixed
		 * and compute statistic. This is done using subsets here */

		/* create index permutation and add as subset to features from p */
		SGVector<index_t>::permute_vector(ind_permutation);

		/* compute statistic for this permutation of mixed samples */
		m_p->add_subset(ind_permutation);
		results[i]=compute_statistic();
		m_p->remove_subset();
	}

	SG_DEBUG("leaving!\n")
	return results;
}

void CIndependenceTest::set_p(CFeatures* p)
{
	/* ref before unref to avoid problems when instances are equal */
	SG_REF(p);
	SG_UNREF(m_p);
	m_p=p;
}

void CIndependenceTest::set_q(CFeatures* q)
{
	/* ref before unref to avoid problems when instances are equal */
	SG_REF(q);
	SG_UNREF(m_q);
	m_q=q;
}

CFeatures* CIndependenceTest::get_p()
{
	SG_REF(m_p);
	return m_p;
}

CFeatures* CIndependenceTest::get_q()
{
	SG_REF(m_q);
	return m_q;
}

