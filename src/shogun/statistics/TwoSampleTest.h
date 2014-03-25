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

#ifndef TWO_SAMPLE_TEST_H_
#define TWO_SAMPLE_TEST_H_

#include <shogun/lib/config.h>
#include <shogun/statistics/HypothesisTest.h>

namespace shogun
{

class CFeatures;

/** @brief Provides an interface for performing the classical two-sample test
 * i.e. Given samples from two distributions \f$p\f$ and \f$q\f$, the
 * null-hypothesis is: \f$H_0: p=q\f$, the alternative hypothesis:
 * \f$H_1: p\neq q\f$.
 *
 * Abstract base class. Provides all interfaces and implements approximating
 * the null distribution via permutation, i.e. repeatedly merging both samples
 * and them compute the test statistic on them.
 *
 */
class CTwoSampleTest : public CHypothesisTest
{
public:
	/** default constructor */
	CTwoSampleTest();

	/** Constructor
	 *
	 * @param p_and_q feature data. Is assumed to contain samples from both
	 * p and q. First all samples from p, then from index m all
	 * samples from q
	 *
	 * @param p_and_q samples from p and q, appended
	 * @param m index of first sample of q
	 */
	CTwoSampleTest(CFeatures* p_and_q, index_t m);

	/** Constructor.
	 * This is a convienience constructor which copies both features to one
	 * element and then calls the other constructor. Needs twice the memory
	 * for a short time
	 *
	 * @param p samples from distribution p, will be copied and NOT
	 * SG_REF'ed
	 * @param q samples from distribution q, will be copied and NOT
	 * SG_REF'ed
	 */
	CTwoSampleTest(CFeatures* p, CFeatures* q);

	/** destructor */
	virtual ~CTwoSampleTest();

	/** merges both sets of samples and computes the test statistic
	 * m_num_permutation_iteration times
	 *
	 * @return vector of all statistics
	 */
	virtual SGVector<float64_t> sample_null();

	/** computes a p-value based on current method for approximating the
	 * null-distribution. The p-value is the 1-p quantile of the null-
	 * distribution where the given statistic lies in.
	 *
	 * @param statistic statistic value to compute the p-value for
	 * @return p-value parameter statistic is the (1-p) percentile of the
	 * null distribution
	 */
	virtual float64_t compute_p_value(float64_t statistic);

	/** computes a threshold based on current method for approximating the
	 * null-distribution. The threshold is the argument of the \f$1-\alpha\f$
	 * quantile of the null. \f$\alpha\f$ is provided.
	 *
	 * @param alpha \f$\alpha\f$ quantile to get the threshold for
	 * @return threshold which is the \f$1-\alpha\f$ quantile of the null
	 * distribution
	 */
	virtual float64_t compute_threshold(float64_t alpha);

	/** Setter for joint features
	 * @param p_and_q joint features from p and q to set
	 */
	virtual void set_p_and_q(CFeatures* p_and_q);

	/** Getter for joint features, SG_REF'ed
	 * @return joint feature object
	 */
	virtual CFeatures* get_p_and_q();

	/** @param m number of samples from first distribution p */
	void set_m(index_t m);

	/** @return number of to be used samples m */
	index_t get_m() { return m_m; }

	virtual const char* get_name() const=0;

private:
	void init();

protected:
	/** concatenated samples of the two distributions (two blocks) */
	CFeatures* m_p_and_q;

	/** defines the first index of samples of q */
	index_t m_m;
};

}

#endif /* TWO_SAMPLE_TEST_H_ */
