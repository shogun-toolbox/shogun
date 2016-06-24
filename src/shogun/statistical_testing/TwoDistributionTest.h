/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2014 - 2016 Soumyajit De
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

#ifndef TWO_DISTRIBUTION_TEST_H_
#define TWO_DISTRIBUTION_TEST_H_

#include <shogun/statistical_testing/HypothesisTest.h>

namespace shogun
{

class CDistance;
class CCustomDistance;

class CTwoDistributionTest : public CHypothesisTest
{
public:
	CTwoDistributionTest();
	virtual ~CTwoDistributionTest();

	void set_p(CFeatures* samples_from_p);
	CFeatures* get_p() const;

	void set_q(CFeatures* samples_from_q);
	CFeatures* get_q() const;

	void set_num_samples_p(index_t num_samples_from_p);
	const index_t get_num_samples_p() const;

	void set_num_samples_q(index_t num_samples_from_q);
	const index_t get_num_samples_q() const;

	CCustomDistance* compute_distance(CDistance* distance);
	CCustomDistance* compute_joint_distance(CDistance* distance);

	virtual float64_t compute_statistic()=0;
	virtual SGVector<float64_t> sample_null()=0;

	virtual const char* get_name() const;
};

}
#endif // TWO_DISTRIBUTION_TEST_H_
