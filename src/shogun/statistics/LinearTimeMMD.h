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

#ifndef LINEAR_TIME_MMD_H_
#define LINEAR_TIME_MMD_H_

#include <shogun/statistics/StreamingMMD.h>

namespace shogun
{

class CStreamingFeatures;
class CFeatures;

/** @brief This class implements the linear time Maximum Mean Statistic as
 * described in [1] for streaming data (see CStreamingMMD for description).
 *
 * Given two sets of samples \f$\{x_i\}_{i=1}^m\sim p\f$ and
 * \f$\{y_i\}_{i=1}^m\sim q\f$
 * the (unbiased) statistic is computed as
 * \f[
 * \text{MMD}_l^2[\mathcal{F},X,Y]=\frac{1}{m_2}\sum_{i=1}^{m_2}
 * h(z_{2i},z_{2i+1})
 * \f]
 * where
 * \f[
 * h(z_{2i},z_{2i+1})=k(x_{2i},x_{2i+1})+k(y_{2i},y_{2i+1})-k(x_{2i},y_{2i+1})-
 * k(x_{2i+1},y_{2i})
 * \f]
 * and \f$ m_2=\lfloor\frac{m}{2} \rfloor\f$.
 *
 * [1]: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B.,
 * & Smola, A. (2012). A Kernel Two-Sample Test. Journal of Machine Learning
 * Research, 13, 671-721.
 */
class CLinearTimeMMD: public CStreamingMMD
{
public:
	/** default constructor */
	CLinearTimeMMD();

	/** Constructor.
	 * @param kernel kernel to use
	 * @param p streaming features p to use
	 * @param q streaming features q to use
	 * @param m number of samples from each distribution
	 * @param blocksize size of examples that are processed at once when
	 * computing statistic/threshold. If larger than m/2, all examples will be
	 * processed at once. Memory consumption increased linearly in the
	 * blocksize. Choose as large as possible regarding available memory.
	 */
	CLinearTimeMMD(CKernel* kernel, CStreamingFeatures* p,
			CStreamingFeatures* q, index_t m, index_t blocksize=10000);

	/** destructor */
	virtual ~CLinearTimeMMD();

	/** Computes squared MMD and a variance estimate, in linear time.
	 * If multiple_kernels is set to true, each subkernel is evaluated on the
	 * same data.
	 *
	 * @param statistic return parameter for statistic, vector with entry for
	 * each kernel. May be allocated before but doesn not have to be
	 *
	 * @param variance return parameter for statistic, vector with entry for
	 * each kernel. May be allocated before but doesn not have to be
	 *
	 * @param multiple_kernels optional flag, if set to true, it is assumed that
	 * the underlying kernel is of type K_COMBINED. Then, the MMD is computed on
	 * all subkernel separately rather than computing it on the combination.
	 * This is used by kernel selection strategies that need to evaluate
	 * multiple kernels on the same data. Since the linear time MMD works on
	 * streaming data, one cannot simply compute MMD, change kernel since data
	 * would be different for every kernel.
	 */
	virtual void compute_statistic_and_variance(
			SGVector<float64_t>& statistic, SGVector<float64_t>& variance,
			bool multiple_kernels=false);

	/** Same as compute_statistic_and_variance, but computes a linear time
	 * estimate of the covariance of the multiple-kernel-MMD.
	 * See [1] for details.
	 */
	virtual void compute_statistic_and_Q(
			SGVector<float64_t>& statistic, SGMatrix<float64_t>& Q);

	/** returns the statistic type of this test statistic */
	virtual EStatisticType get_statistic_type() const
	{
		return S_LINEAR_TIME_MMD;
	}

	/** @return the class name */
	virtual const char* get_name() const
	{
		return "LinearTimeMMD";
	}

protected:
	/** method that computes the squared MMD in linear time (see class
	 * description for the equation)
	 *
	 * @param kernel the kernel to be used for computing MMD. This will be
	 * useful when multiple kernels are used
	 * @param data the list of data on which kernels are computed. The order
	 * of data in the list is \f$x,x',\cdots\sim p\f$ followed by
	 * \f$y,y',\cdots\sim q\f$. It is assumed that detele_data flag is set
	 * inside the list
	 * @param num_this_run number of data points in current blocks
	 * @return the MMD values (the h-vectors)
	 */
	 virtual SGVector<float64_t> compute_squared_mmd(CKernel* kernel,
			 CList* data, index_t num_this_run);

private:
	/** helper method, same as compute_squared_mmd with an option to use
	 * preallocated memory for faster processing */
	void compute_squared_mmd(CKernel* kernel, CList* data,
			SGVector<float64_t>& current, SGVector<float64_t>& pp,
			SGVector<float64_t>& qq, SGVector<float64_t>& pq,
			SGVector<float64_t>& qp, index_t num_this_run);

};

}

#endif /* LINEAR_TIME_MMD_H_ */

