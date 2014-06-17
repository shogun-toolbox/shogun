/*
 * Copyright (c) The Shogun Machine Learning Toolbox
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

#ifndef BTEST_MMD_H_
#define BTEST_MMD_H_

#include <shogun/lib/config.h>

#include <shogun/statistics/StreamingMMD.h>

namespace shogun
{

/** @brief This class implements the block-based Maximum Mean Statistic as
 * described in [1] for streaming data (see CStreamingMMD for background
 * description).
 *
 * In particular, Given two sets of samples \f$\{x_i\}_{i=1}^{n_x}\sim p\f$ and
 * \f$\{y_i\}_{i=1}^{n_y}\sim q\f$ with block-size \f$B\f$ kept fixed, the
 * unbiased estimate of the test statistic under \f$\mathbf{H}_0\f$ follows the
 * following asymptotic behavior
 * \f[
 *	\frac{n_xn_y}{n_x+n_y}\sqrt{\frac{B}{n}}\hat{\eta}_k\rightarrow
 *	\mathcal{N}\left(0,\sigma_{k,0}^2\right)
 * \f]
 * as \f$n\rightarrow\infty\f$.
 *
 * This class just defines the statistic multiplier \f$\zeta\f$ and \f$\gamma\f$
 * as \f$\frac{n_xn_y}{n_x+n_y}\sqrt{\frac{B}{n}}\f$ and \f$\left(\frac{B_xB_y}
 * {B_x+B_y} \right )^2\f$ respectively. The multiplier for Gaussian
 * approximation in this case is just 1 (see CLinearTimeMMD)
 *
 * [1] W. Zaremba, A. Gretton, and M. Blaschko, B-test: A Non-parametric,
 * Low Variance Kernel Two-sample Test. In Advances in Neural Information
 * Processing Systems (NIPS), 2013.
 */
class CBTestMMD: public CStreamingMMD
{
public:
	/** default constructor */
	CBTestMMD();

	/** Constructor.
	 * @param kernel kernel to use
	 * @param p streaming features p to use
	 * @param q streaming features q to use
	 * @param m number of samples from both distributions
	 * @param blocksize size of examples that are processed at once when
	 * computing statistic/threshold. Default is 250.
	 */
	CBTestMMD(CKernel* kernel, CStreamingFeatures* p, CStreamingFeatures* q,
			index_t m, index_t blocksize=250);

	/** Constructor.
	 * @param kernel kernel to use
	 * @param p streaming features p to use
	 * @param q streaming features q to use
	 * @param m number of samples from first distribution, p
	 * @param n number of samples from first distribution, q
	 * @param blocksize size of examples that are processed at once when
	 * computing statistic/threshold.
	 */
	CBTestMMD(CKernel* kernel, CStreamingFeatures* p, CStreamingFeatures* q,
			index_t m, index_t n, index_t blocksize);

	/** destructor */
	virtual ~CBTestMMD();

	/** returns the statistic type of this test statistic */
	virtual EStatisticType get_statistic_type() const
	{
		return S_BTEST_MMD;
	}

	/** @return the class name */
	virtual const char* get_name() const
	{
		return "BTestMMD";
	}

protected:
	/** Method that computes statistic estimate multiplier \f$\zeta\f$ as
	 * \f$\frac{n_xn_y}{n_x+n_y}\sqrt{\frac{B}{n}}\f$
	 *
	 * @return multiplier \f$\zeta\f$ for statistic estimate
	 */
	virtual float64_t compute_stat_est_multiplier();

	/** Method that computes variance estimate multiplier \f$\gamma\f$ for
	 * within block permuation approach as \f$\left(\frac{B_xB_y}{B_x+B_y}
	 * \right )^2\f$.
	 *
	 * @return multiplier \f$\gamma\f$ for variance estimate under null
	 */
	virtual float64_t compute_var_est_multiplier();

	/** Method that computes multiplier for Gaussian approximation of
	 * asymptotic distribution under null. For B-test this just returns 1
	 *
	 * @return multiplier for Gaussian approximation
	 */
	virtual float64_t compute_gaussian_multiplier();
};

}

#endif /* BTEST_MMD_H_ */

