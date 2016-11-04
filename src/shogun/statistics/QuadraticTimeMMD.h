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

#ifndef QUADRATIC_TIME_MMD_H_
#define QUADRATIC_TIME_MMD_H_

#include <shogun/lib/config.h>

#include <shogun/statistics/KernelTwoSampleTest.h>

namespace shogun
{

class CFeatures;
class CKernel;
class CCustomKernel;

/** Enum to select which statistic type of quadratic time MMD should be computed */
enum EQuadraticMMDType
{
	BIASED,
	BIASED_DEPRECATED,
	UNBIASED,
	UNBIASED_DEPRECATED,
	INCOMPLETE
};

/** @brief This class implements the quadratic time Maximum Mean Statistic as
 * described in [1].
 * The MMD is the distance of two probability distributions \f$p\f$ and \f$q\f$
 * in a RKHS which we denote by
 * \f[
 * 	\hat{\eta_k}=\text{MMD}[\mathcal{F},p,q]^2=\textbf{E}_{x,x'}
 * 	\left[ k(x,x')\right]-2\textbf{E}_{x,y}\left[ k(x,y)\right]
 * 	+\textbf{E}_{y,y'}\left[ k(y,y')\right]=||\mu_p - \mu_q||^2_\mathcal{F}
 * \f]
 *
 * Given two sets of samples \f$\{x_i\}_{i=1}^{n_x}\sim p\f$ and
 * \f$\{y_i\}_{i=1}^{n_y}\sim q\f$, \f$n_x+n_y=n\f$,
 * the unbiased estimate of the above statistic is computed as
 * \f[
 * 	\hat{\eta}_{k,U}=\frac{1}{n_x(n_x-1)}\sum_{i=1}^{n_x}\sum_{j\neq i}
 * 	k(x_i,x_j)+\frac{1}{n_y(n_y-1)}\sum_{i=1}^{n_y}\sum_{j\neq i}k(y_i,y_j)
 * 	-\frac{2}{n_xn_y}\sum_{i=1}^{n_x}\sum_{j=1}^{n_y}k(x_i,y_j)
 * \f]
 *
 * A biased version is
 * \f[
 * 	\hat{\eta}_{k,V}=\frac{1}{n_x^2}\sum_{i=1}^{n_x}\sum_{j=1}^{n_x}
 * 	k(x_i,x_j)+\frac{1}{n_y^2}\sum_{i=1}^{n_y}\sum_{j=1}^{n_y}k(y_i,y_j)
 * 	-\frac{2}{n_xn_y}\sum_{i=1}^{n_x}\sum_{j=1}^{n_y}k(x_i,y_j)
 * \f]
 *
 * When \f$n_x=n_y=\frac{n}{2}\f$, an incomplete version can also be computed
 * as the following
 * \f[
 * 	\hat{\eta}_{k,U^-}=\frac{1}{\frac{n}{2}(\frac{n}{2}-1)}\sum_{i\neq j}
 * 	h(z_i,z_j)
 * \f]
 * where for each pair \f$z=(x,y)\f$, \f$h(z,z')=k(x,x')+k(y,y')-k(x,y')-
 * k(x',y)\f$.
 *
 * The type (biased/unbiased/incomplete) can be selected via set_statistic_type().
 * Note that there are presently two setups for computing statistic. While using
 * BIASED, UNBIASED or INCOMPLETE, the estimate returned by compute_statistic()
 * is \f$\frac{n_xn_y}{n_x+n_y}\hat{\eta}_k\f$. If DEPRECATED ones are used, then
 * this returns \f$(n_x+n_y)\hat{\eta}_k\f$ in general and \f$(\frac{n}{2})
 * \hat{\eta}_k\f$ when \f$n_x=n_y=\frac{n}{2}\f$. This holds for the null
 * distribution samples as well.
 *
 * Estimating variance of the asymptotic distribution of the statistic under
 * null and alternative hypothesis can be done using compute_variance() method.
 * This is internally done alongwise computing statistics to avoid recomputing
 * the kernel.
 *
 * Variance under null is computed as
 * \f$\sigma_{k,0}^2=2\hat{\kappa}_2=2(\kappa_2-2\kappa_1+\kappa_0)\f$
 * where
 * \f$\kappa_0=\left(\mathbb{E}_{X,X'}k(X,X')\right )^2\f$,
 * \f$\kappa_1=\mathbb{E}_X\left[(\mathbb{E}_{X'}k(X,X'))^2\right]\f$, and
 * \f$\kappa_2=\mathbb{E}_{X,X'}k^2(X,X')\f$
 * and variance under alternative is computed as
 * \f[
 * 	\sigma_{k,A}^2=4\rho_y\left\{\mathbb{E}_X\left[\left(\mathbb{E}_{X'}
 * 	k(X,X')-\mathbb{E}_Yk(X,Y)\right)^2 \right ] -\left(\mathbb{E}_{X,X'}
 * 	k(X,X')-\mathbb{E}_{X,Y}k(X,Y) \right)^2\right \}+4\rho_x\left\{
 * 	\mathbb{E}_Y\left[\left(\mathbb{E}_{Y'}k(Y,Y')-\mathbb{E}_Xk(X,Y)
 * 	\right)^2\right ] -\left(\mathbb{E}_{Y,Y'}k(Y,Y')-\mathbb{E}_{X,Y}
 * 	k(X,Y) \right)^2\right \}
 * \f]
 * where \f$\rho_x=\frac{n_x}{n}\f$ and \f$\rho_y=\frac{n_y}{n}\f$.
 *
 * Note that statistic and variance estimation can be done for multiple kernels
 * at once as well.
 *
 * Along with the statistic comes a method to compute a p-value based on
 * different methods. Permutation test is also possible. If unsure which one to
 * use, sampling with 250 permutation iterations always is correct (but slow).
 *
 * To choose, use set_null_approximation_method() and choose from.
 *
 * MMD2_SPECTRUM_DEPRECATED: For a fast, consistent test based on the spectrum of
 * the kernel matrix, as described in [2]. Only supported if Eigen3 is installed.
 *
 * MMD2_SPECTRUM: Similar to the deprecated version except it estimates the
 * statistic under null as \f$\frac{n_xn_y}{n_x+n_y}\hat{\eta}_{k,U}\rightarrow
 * \sum_r\lambda_r(Z_r^2-1)\f$ instead (see method description for more details).
 *
 * MMD2_GAMMA: for a very fast, but not consistent test based on moment matching
 * of a Gamma distribution, as described in [2].
 *
 * PERMUTATION: For permuting available samples to sample null-distribution
 *
 * If you do not know about your data, but want to use the MMD from a kernel
 * matrix, just use the custom kernel constructor. Everything else will work as
 * usual.
 *
 * For kernel selection see CMMDKernelSelection.
 *
 * NOTE: \f$n_x\f$ and \f$n_y\f$ are represented by \f$m\f$ and \f$n\f$,
 * respectively in the implementation.
 *
 * [1]: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B., & Smola, A. (2012).
 * A Kernel Two-Sample Test. Journal of Machine Learning Research, 13, 671-721.
 *
 * [2]: Gretton, A., Fukumizu, K., & Harchaoui, Z. (2011).
 * A fast, consistent kernel two-sample test.
 *
 */
class CQuadraticTimeMMD : public CKernelTwoSampleTest
{
public:
	/** default constructor */
	CQuadraticTimeMMD();

	/** Constructor
	 *
	 * @param p_and_q feature data. Is assumed to contain samples from both
	 * p and q. First m samples from p, then from index m all samples from q
	 *
	 * @param kernel kernel to use
	 * @param p_and_q samples from p and q, appended
	 * @param m index of first sample of q
	 */
	CQuadraticTimeMMD(CKernel* kernel, CFeatures* p_and_q, index_t m);

	/** Constructor.
	 * This is a convienience constructor which copies both features to one
	 * element and then calls the other constructor. Needs twice the memory
	 * for a short time
	 *
	 * @param kernel kernel for MMD
	 * @param p samples from distribution p, will be copied and NOT SG_REF'ed
	 * @param q samples from distribution q, will be copied and NOT SG_REF'ed
	 */
	CQuadraticTimeMMD(CKernel* kernel, CFeatures* p, CFeatures* q);

	/** Constructor.
	 * This is a convienience constructor which allows to only specify
	 * a custom kernel. In this case, the features are completely ignored
	 * and all computations will be done on the custom kernel
	 *
	 * @param custom_kernel custom kernel for MMD, which is a kernel between
	 * the appended features p and q
	 * @param m index of first sample of q
	 */
	CQuadraticTimeMMD(CCustomKernel* custom_kernel, index_t m);

	/** destructor */
	virtual ~CQuadraticTimeMMD();

	/** Computes the squared quadratic time MMD for the current data. Note
	 * that the type (biased/unbiased/incomplete) can be specified with
	 * set_statistic_type() method.
	 *
	 * @return (biased, unbiased or incomplete) \f$\frac{mn}{m+n}\hat{\eta}_k\f$.
	 * If DEPRECATED types are used, then it returns \f$(m+m)\hat{\eta}_k\f$ in
	 * general and \f$m\hat{\eta}_k\f$ when \f$m=n\f$.
	 */
	virtual float64_t compute_statistic();

	/** Same as compute_statistic(), but with the possibility to perform on
	 * multiple kernels at once
	 *
	 * @param multiple_kernels if true, and underlying kernel is K_COMBINED,
	 * method will be executed on all subkernels on the same data
	 * @return vector of results for subkernels
	 */
	SGVector<float64_t> compute_statistic(bool multiple_kernels);

	/**
	 * Wrapper for computing variance estimate of the asymptotic distribution
	 * of the statistic (unbisaed/biased/incomplete) under null and alternative
	 * hypothesis (see class description for details)
	 *
	 * @return a vector of two values containing asymptotic variance estimate
	 * under null and alternative, respectively
	 */
	virtual SGVector<float64_t> compute_variance();

	/** Same as compute_variance(), but with the possibility to perform on
	 * multiple kernels at once
	 *
	 * @param multiple_kernels if true, and underlying kernel is K_COMBINED,
	 * method will be executed on all subkernels on the same data
	 * @return matrix of results for subkernels, one row for each subkernel
	 */
	SGMatrix<float64_t> compute_variance(bool multiple_kernels);

	/**
	 * Wrapper method for compute_variance()
	 *
	 * @return variance estimation of asymptotic distribution of statistic
	 * under null hypothesis
	 */
	float64_t compute_variance_under_null();

	/**
	 * Wrapper method for compute_variance()
	 *
	 * @return variance estimation of asymptotic distribution of statistic
	 * under alternative hypothesis
	 */
	float64_t compute_variance_under_alternative();

	/** computes a p-value based on current method for approximating the
	 * null-distribution. The p-value is the 1-p quantile of the null-
	 * distribution where the given statistic lies in.
	 *
	 * Not all methods for computing the p-value are compatible with all
	 * methods of computing the statistic (biased/unbiased/incomplete).
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
	 * Not all methods for computing the p-value are compatible with all
	 * methods of computing the statistic (biased/unbiased/incomplete).
	 *
	 * @param alpha test level to reject null-hypothesis
	 * @return threshold for statistics to reject null-hypothesis
	 */
	virtual float64_t compute_threshold(float64_t alpha);

	/** @return the class name */
	virtual const char* get_name() const
	{
		return "QuadraticTimeMMD";
	};

	/** returns the statistic type of this test statistic */
	virtual EStatisticType get_statistic_type() const
	{
		return S_QUADRATIC_TIME_MMD;
	}

	/** Returns a set of samples of an estimate of the null distribution
	 * using the Eigen-spectrum of the centered kernel matrix of the merged
	 * samples of p and q. May be used to compute p-value (easy).
	 *
	 * The estimate is computed as
	 * \f[
	 *	\frac{n_xn_y}{n_x+n_y}\hat{\eta}_{k,U}\rightarrow\sum_{l=1}^\infty
	 *	\lambda_l\left(Z^2_l-1 \right)
	 * \f]
	 * where \f${Z_l}\stackrel{i.i.d.}{\sim}\mathcal{N}(0,1)\f$ and
	 * \f$\lambda_l\f$ are the eigenvalues of centered kernel matrix HKH.
	 *
	 * kernel matrix needs to be stored in memory
	 *
	 * Note that m*n/(m+n)*Null-distribution is returned,
	 * which is fine since the statistic is also m*n/(m+n)*MMD^2
	 *
	 * Works well if the kernel matrix is NOT diagonal dominant.
	 * See Gretton, A., Fukumizu, K., & Harchaoui, Z. (2011).
	 * A fast, consistent kernel two-sample test.
	 *
	 * @param num_samples number of samples to draw
	 * @param num_eigenvalues number of eigenvalues to use to draw samples
	 * Maximum number of m+n-1 where m and n are the sizes of samples from
	 * p and q respectively.
	 * @return samples from the estimated null distribution
	 */
	SGVector<float64_t> sample_null_spectrum(index_t num_samples,
			index_t num_eigenvalues);

	/** Returns a set of samples of an estimate of the null distribution
	 * using the Eigen-spectrum of the centered kernel matrix of the merged
	 * samples of p and q. May be used to compute p-value (easy).
	 *
	 * The unbiased version uses
	 * \f[
	 *	t\text{MMD}_u^2[\mathcal{F},X,Y]\rightarrow\sum_{l=1}^\infty
	 *	\lambda_l\left((a_l\rho_x^{-\frac{1}{{2}}}
	 *	-b_l\rho_y^{-\frac{1}{{2}}})^2-(\rho_x\rho_y)^{-1} \right)
	 * \f]
	 * where \f$t=m+n\f$, \f$\lim_{m,n\rightarrow\infty}m/t\rightarrow
	 * \rho_x\f$ and \f$\rho_y\f$ likewise (equation 10 from [1]) and
	 * \f$\lambda_l\f$ are estimated as \f$\frac{\nu_l}{(m+n)}\f$, where
	 * \f$\nu_l\f$ are the eigenvalues of centered kernel matrix HKH.
	 *
	 * The biased version uses
	 * \f[
	 * 	t\text{MMD}_b^2[\mathcal{F},X,Y]\rightarrow\sum_{l=1}^\infty
	 *	\lambda_l\left((a_l\rho_x^{-\frac{1}{{2}}}-
	 *	b_l\rho_y^{-\frac{1}{{2}}})^2\right)
	 * \f]
	 *
	 * kernel matrix needs to be stored in memory
	 *
	 * Note that (m+n)*Null-distribution is returned,
	 * which is fine since the statistic is also (m+n)*MMD:
	 * except when m and n are equal, then m*MMD^2 is returned
	 *
	 * Works well if the kernel matrix is NOT diagonal dominant.
	 * See Gretton, A., Fukumizu, K., & Harchaoui, Z. (2011).
	 * A fast, consistent kernel two-sample test.
	 *
	 * @param num_samples number of samples to draw
	 * @param num_eigenvalues number of eigenvalues to use to draw samples
	 * Maximum number of m+n-1 where m and n are the sizes of samples from
	 * p and q respectively.
	 * It is usually safe to use a smaller number since they decay very
	 * fast, however, a conservative approach would be to use all (-1 does
	 * this). See paper for details.
	 * @return samples from the estimated null distribution
	 */
	SGVector<float64_t> sample_null_spectrum_DEPRECATED(index_t num_samples,
			index_t num_eigenvalues);

	/** setter for number of samples to use in spectrum based p-value
	 * computation.
	 *
	 * @param num_samples_spectrum number of samples to draw from
	 * approximate null-distributrion
	 */
	void set_num_samples_spectrum(index_t num_samples_spectrum);

	/** setter for number of eigenvalues to use in spectrum based p-value
	 * computation. Maximum is m_m+m_n-1
	 *
	 * @param num_eigenvalues_spectrum number of eigenvalues to use to
	 * approximate null-distributrion
	 */
	void set_num_eigenvalues_spectrum(index_t num_eigenvalues_spectrum);

	/** @param statistic_type statistic type (biased/unbiased/incomplete) to use */
	void set_statistic_type(EQuadraticMMDType statistic_type);

	/** Approximates the null-distribution by the two parameter gamma
	 * distribution. It works in O(m^2) where m is the number of samples
	 * from each distribution. Its very fast, but may be inaccurate.
	 * However, there are cases where it performs very well.
	 * Returns parameters of gamma distribution that is fitted.
	 *
	 * Called by compute_p_value() if null approximation method is set to
	 * MMD2_GAMMA.
	 *
	 * Note that when being used for constructing a test, the provided
	 * statistic HAS to be the biased version (see paper for details). To use,
	 * set BIASED_DEPRECATED as statistic type. Note that m*Null-distribution
	 * is fitted, which is fine since the statistic is also m*MMD.
	 *
	 * See Gretton, A., Fukumizu, K., & Harchaoui, Z. (2011).
	 * A fast, consistent kernel two-sample test.
	 *
	 * @return vector with two parameter for gamma distribution. To use:
	 * call gamma_cdf(statistic, a, b).
	 */
	SGVector<float64_t> fit_null_gamma();

protected:
	/**
	 * Helper method to compute unbiased estimate of squared quadratic time MMD
	 * and variance estimate under null and alternative hypothesis
	 *
	 * @param m number of samples from p
	 * @param n number of samples from q
	 * @return a vector of three values
	 * first - unbiased \f$\text{MMD}^2\f$ estimate \f$\hat{\eta}_{k,U}\f$
	 * second - variance under null hypothesis (see class documentation)
	 * third - variance under alternative hypothesis (see class documentation)
	 */
	SGVector<float64_t> compute_unbiased_statistic_variance(int m, int n);

	/**
	 * Helper method to compute biased estimate of squared quadratic time MMD
	 * and variance estimate under null and alternative hypothesis
	 *
	 * @param m number of samples from p
	 * @param n number of samples from q
	 * @return a vector of three values
	 * first - biased \f$\text{MMD}^2\f$ estimate \f$\hat{\eta}_{k,V}\f$
	 * second - variance under null hypothesis (see class documentation)
	 * third - variance under alternative hypothesis (see class documentation)
	 */
	SGVector<float64_t> compute_biased_statistic_variance(int m, int n);

	/**
	 * Helper method to compute incomplete estimate of squared quadratic time MMD
	 * and variance estimate under null and alternative hypothesis
	 *
	 * @param n number of samples from p and q
	 * @return a vector of three values
	 * first - incomplete \f$\text{MMD}^2\f$ estimate \f$\hat{\eta}_{k,U^-}\f$
	 * second - variance under null hypothesis (see class documentation)
	 * third - variance under alternative hypothesis (see class documentation)
	 */
	SGVector<float64_t> compute_incomplete_statistic_variance(int n);

	/** Wrapper method for computing unbiased estimate of MMD^2
	 *
	 * @param m number of samples from p
	 * @param n number of samples from q
	 * @return unbiased \f$\text{MMD}^2\f$ estimate \f$\hat{\eta}_{k,U}\f$
	 */
	float64_t compute_unbiased_statistic(int m, int n);

	/** Wrapper method for computing biased estimate of MMD^2
	 *
	 * @param m number of samples from p
	 * @param n number of samples from q
	 * @return biased \f$\text{MMD}^2\f$ estimate \f$\hat{\eta}_{k,V}\f$
	 */
	float64_t compute_biased_statistic(int m, int n);

	/** Wrapper method for computing incomplete estimate of MMD^2
	 *
	 * @param n number of samples from p and q
	 * @return incomplete \f$\text{MMD}^2\f$ estimate \f$\hat{\eta}_{k,U^-}\f$
	 */
	float64_t compute_incomplete_statistic(int n);

private:
	/** register parameters and initialize with defaults */
	void init();

protected:
	/** number of samples for spectrum null-dstribution-approximation */
	index_t m_num_samples_spectrum;

	/** number of Eigenvalues for spectrum null-dstribution-approximation */
	index_t m_num_eigenvalues_spectrum;

	/** type of statistic (biased/unbiased/incomplete as well as deprecated
	 * versions of biased/unbiased)
	 */
	EQuadraticMMDType m_statistic_type;
};

}

#endif /* QUADRATIC_TIME_MMD_H_ */
