/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#ifndef __HSIC_H_
#define __HSIC_H_

#include <shogun/statistics/KernelIndependenceTestStatistic.h>

namespace shogun
{

template<class T> class SGMatrix;


/** @brief This class implements the Hilbert Schmidtd Independence Criterion
 * based independence test as described in [1].
 *
 * Given samples \f$Z=\{(x_i,y_i)\}_{i=1}^m\f$ from the joint
 * distribution \f$\textbf{P}_{xy}\f$, does the joint distribution
 * factorize as \f$\textbf{P}_{xy}=\textbf{P}_x\textbf{P}_y\f$?
 *
 * The HSIC is a kernel based independence criterion, which is based on the
 * largest singular value of a Cross-Covariance Operator in a reproducing
 * kernel Hilbert space (RKHS). Its population expression is zero if and only
 * if the two underlying distributions are independent.
 *
 * This class can compute empirical biased estimates:
 * \f[
 * m\text{HSIC}(Z)[,p,q]^2)=\frac{1}{m^2}\text{trace}\textbf{KHLH}
 * \f]
 * where \f$\textbf{H}=\textbf{I}-\frac{1}{m}\textbf{11}^T\f$ is a centering
 * matrix and \f$\textbf{K}, \textbf{L}\f$ are kernel matrices of both sets
 * of samples.
 *
 * Note that computing the statistic returns m*MMD; same holds for the null
 * distribution samples.
 *
 * Along with the statistic comes a method to compute a p-value based on
 * different methods. Sampling from null is also possible. If unsure which one to
 * use, sampling with 250 iterations always is correct (but slow).
 *
 * To choose, use set_null_approximation_method() and choose from
 *
 * HSIC_GAMMA: for a very fast, but not consistent test based on moment matching
 * of a Gamma distribution, as described in [1].
 *
 * PERMUTATION: For permuting available samples to sample null-distribution.
 * This is done on precomputed kernel matrices, since they have to
 * be stored anyway when the statistic is computed.
 *
 * A very basic method for kernel selection when using CGaussianKernel is to
 * use the median distance of the underlying data. See examples how to do that.
 * More advanced methods will follow in the near future. However, the median
 * heuristic works in quite some cases. See [1].
 *
 * [1]: Gretton, A., Fukumizu, K., Teo, C., & Song, L. (2008).
 * A kernel statistical test of independence.
 * Advances in Neural Information Processing Systems, 1-8.
 *
 */
class CHSIC : public CKernelIndependenceTestStatistic
{
public:
	/** Constructor */
	CHSIC();

	/** Constructor
	 *
	 * @param p_and_q feature data. Is assumed to contain samples from both
	 * p and q. First all samples from p, then from index m all
	 * samples from q
	 *
	 * @param kernel_p kernel to use on samples from p
	 * @param kernel_q kernel to use on samples from q
	 * @param p_and_q samples from p and q, appended
	 * @param m index of first sample of q
	 */
	CHSIC(CKernel* kernel_p, CKernel* kernel_q, CFeatures* p_and_q,
			index_t m);

	/** Constructor.
	 * This is a convienience constructor which copies both features to one
	 * element and then calls the other constructor. Needs twice the memory
	 * for a short time
	 *
	 * @param kernel_p kernel to use on samples from p
	 * @param kernel_q kernel to use on samples from q
	 * @param p samples from distribution p, will be copied and NOT
	 * SG_REF'ed
	 * @param q samples from distribution q, will be copied and NOT
	 * SG_REF'ed
	 */
	CHSIC(CKernel* kernel_p, CKernel* kernel_q, CFeatures* p, CFeatures* q);

	virtual ~CHSIC();

	/** Computes the HSIC statistic (see class description) for underlying
	 * kernels and data. Note that it is multiplied by the number of used
	 * samples. It is a biased estimator. Note that it is m*HSIC_b.
	 *
	 * Note that since kernel matrices have to be stored, it has quadratic
	 * space costs.
	 *
	 * @return m*HSIC (unbiased estimate)
	 */
	virtual float64_t compute_statistic();

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
	 * null-distribution. The threshold is the value that a statistic has
	 * to have in ordner to reject the null-hypothesis.
	 *
	 * @param alpha test level to reject null-hypothesis
	 * @return threshold for statistics to reject null-hypothesis
	 */
	virtual float64_t compute_threshold(float64_t alpha);

	virtual const char* get_name() const
	{
		return "HSIC";
	}

	/** returns the statistic type of this test statistic */
	virtual EStatisticType get_statistic_type() const
	{
		return S_HSIC;
	}

	/** Approximates the null-distribution by a two parameter gamma
	 * distribution. Returns parameters.
	 *
	 * NOTE: the gamma distribution is fitted to m*HSIC_b. But since
	 * compute_statistic() returnes the biased estimate, you can safely call
	 * this with values from compute_statistic().
	 * However, the attached features have to be the SAME size, as these, the
	 * statistic was computed on. If compute_threshold() or compute_p_value()
	 * are used, this is ensured automatically. Note that m*Null-distribution is
	 * fitted, which is fine since the statistic is also m*HSIC.
	 *
	 * Has quadratic computational costs in terms of samples.
	 *
	 * Called by compute_p_value() if null approximation method is set to
	 * MMD2_GAMMA.
	 *
	 * @return vector with two parameters for gamma distribution. To use:
	 * call gamma_cdf(statistic, a, b).
	 */
	SGVector<float64_t> fit_null_gamma();

	/** merges both sets of samples and computes the test statistic
	 * m_num_null_sample times. This version precomputes the kenrel matrix
	 * once by hand, then samples using this one. The matrix has
	 * to be stored anyway when statistic is computed.
	 *
	 * @return vector of all statistics
	 */
	virtual SGVector<float64_t> sample_null();

protected:
	/** @return kernel matrix on samples from p. Distinguishes CustomKernels */
	SGMatrix<float64_t> get_kernel_matrix_K();

	/** @return kernel matrix on samples from q. Distinguishes CustomKernels */
	SGMatrix<float64_t> get_kernel_matrix_L();

private:
	void init();

};

}

#endif /* __HSIC_H_ */
