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

#ifndef INDEPENDENCE_TEST_H_
#define INDEPENDENCE_TEST_H_

#include <shogun/statistics/HypothesisTest.h>

namespace shogun
{

class CFeatures;

/** @brief Provides an interface for performing the independence test.
 * Given samples \f$Z=\{(x_i,y_i)\}_{i=1}^m\f$ from the joint distribution
 * \f$\textbf{P}_{xy}\f$, does the joint distribution factorize as
 * \f$\textbf{P}_{xy}=\textbf{P}_x\textbf{P}_y\f$, i.e. product of the marginals?
 * The null-hypothesis says yes, i.e. no dependence, the alternative hypothesis
 * says no.
 *
 * Abstract base class. Provides all interfaces and implements approximating
 * the null distribution via permutation, i.e. shuffling the samples from
 * one distribution repeatedly using subsets while keeping the samples from
 * the other distribution in its original order
 *
 */
class CIndependenceTest : public CHypothesisTest
{
public:
	/** default constructor */
	CIndependenceTest();

	/** Constructor.
	 *
	 * Initializes the features from the two distributions and SG_REFs them
	 *
	 * @param p samples from distribution p
	 * @param q samples from distribution q
	 */
	CIndependenceTest(CFeatures* p, CFeatures* q);

	/** destructor */
	virtual ~CIndependenceTest();

	/** shuffles samples from one distribution keeping the samples from another
	 * distribution in the original order and computes the test statistic
	 * m_num_null_sample times
	 *
	 * @return vector of all statistics
	 */
	virtual SGVector<float64_t> sample_null();

	/** Setter for features from distribution p, SG_REFs it
	 *
	 * @param p features from p
	 */
	virtual void set_p(CFeatures* p);

	/** Setter for features from distribution q, SG_REFs it
	 *
	 * @param q features from q
	 */
	virtual void set_q(CFeatures* q);

	/** Getter for features from p, SG_REF'ed
	 *
	 * @return feature object from p
	 */
	virtual CFeatures* get_p();

	/** Getter for features from q, SG_REF'ed
	 *
	 * @return feature object from q
	 */
	virtual CFeatures* get_q();

	/** @return class name */
	virtual const char* get_name() const=0;

private:
	/** register parameters and initialize with default values */
	void init();

protected:
	/** samples of the distribution p */
	CFeatures* m_p;

	/** samples of the distribution q */
	CFeatures* m_q;
};

}

#endif /* INDEPENDENCE_TEST_H_ */
