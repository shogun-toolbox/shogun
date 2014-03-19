/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012-2013 Heiko Strathmann
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

#include <shogun/statistics/HypothesisTest.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;

CHypothesisTest::CHypothesisTest() : CSGObject()
{
	init();
}

CHypothesisTest::~CHypothesisTest()
{
}

void CHypothesisTest::init()
{
	SG_ADD(&m_num_null_samples, "num_null_samples",
			"Number of permutation iterations for sampling null",
			MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*)&m_null_approximation_method,
			"null_approximation_method",
			"Method for approximating null distribution",
			MS_NOT_AVAILABLE);

	m_num_null_samples=250;
	m_null_approximation_method=PERMUTATION;
}

void CHypothesisTest::set_null_approximation_method(
		ENullApproximationMethod null_approximation_method)
{
	m_null_approximation_method=null_approximation_method;
}

void CHypothesisTest::set_num_null_samples(index_t num_null_samples)
{
	m_num_null_samples=num_null_samples;
}

float64_t CHypothesisTest::compute_p_value(float64_t statistic)
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

float64_t CHypothesisTest::compute_threshold(float64_t alpha)
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

float64_t CHypothesisTest::perform_test()
{
	/* baseline method here is simply to compute statistic and p-value
	 * separately */
	float64_t statistic=compute_statistic();
	return compute_p_value(statistic);
}

bool CHypothesisTest::perform_test(float64_t alpha)
{
	float64_t p_value=perform_test();
	return p_value<alpha;
}
