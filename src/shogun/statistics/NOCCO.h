/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
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

#ifndef NOCCO_H_
#define NOCCO_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/statistics/KernelIndependenceTest.h>

namespace shogun
{

template<class T> class SGMatrix;

/** @brief This class implements the NOrmalized Cross Covariance Operator
 * (NOCCO) based independence test as described in [1].
 *
 * The test of independence is performed as follows: Given samples \f$Z=\{(x_i,
 * y_i)\}_{i=1}^n\f$ from the joint distribution \f$\textbf{P}_{XY}\f$,
 * does the joint distribution factorize as \f$\textbf{P}_{XY}=\textbf{P}_X
 * \textbf{P}_Y\f$? The null hypothesis says yes and the alternative hypothesis
 * says no.
 *
 * The dependence of the random variables \f$\mathbf X=\{x_i\}\f$ and \f$
 * \mathbf Y=\{y_i\}\f$ can be measured via the cross-covariance operator
 * \f$\boldsymbol\Sigma_{YX}\f$ which becomes \f$\mathbf{0}\f$ if and only if
 * \f$\mathbf X\f$ and \f$\mathbf Y\f$ are independent. This term factorizes as
 * \f$\boldsymbol\Sigma_{YX}=\boldsymbol\Sigma_{YY}^{\frac{1}{2}}\mathbf{V}_
 * {YX}\boldsymbol\Sigma_{XX}^{\frac{1}{2}}\f$, where \f$\boldsymbol\Sigma_
 * {XX}\f$ and \f$\boldsymbol\Sigma_{YY}\f$ are known as covariance operator and
 * \f$\mathbf{V}_{YX}\f$ is known as normalized cross-covariance operator. The
 * paper uses the Hilbert-Schmidt norm of \f$\mathbf V_{YX}\f$ as a dependence
 * measure of the independence test (see paper for theroretical details).
 *
 * This class overrides the compute_statistic() method of the superclass which
 * computes an unbiased estimate of the normalized cross covariance operator
 * norm. Given the kernels \f$K\f$ (for \f$\mathbf X\f$) and \f$L\f$ (for
 * \f$\mathbf Y\f$), if we denote the doubly centered Gram matrices as
 * \f$\mathbf{G}_X=\mathbf{HKH}\f$ and \f$\mathbf{G}_Y=\mathbf{HLH}\f$
 * (where \f$\mathbf H=\mathbf I-\frac{1}{n}\mathbf{1}\f$), then the operator
 * norm is estimated as
 * \f[
 *	\hat{I}^{\text{NOCCO}}=\text{Trace}\left[\mathbf{R_X R_Y}\right]
 * \f]
 * where \f$\mathbf{R}_X=\mathbf{G}_X(\mathbf{G}_X+n\varepsilon_n\mathbf{I})
 * ^{-1}\f$ and \f$\mathbf{R}_Y=\mathbf{G}_Y(\mathbf{G}_Y+n\varepsilon_n
 * \mathbf{I})^{-1}\f$ and \f$\varepsilon_n\gt 0\f$ is a regularization
 * constant.
 *
 * In order to avoid computing direct inverse in the above terms for avoiding
 * numerical issues, this class uses Cholesky decomposition of matrices
 * \f$\mathbf{GG}_*=\mathbf{LL}^\top\f$ (where \f$\mathbf{GG}_*=(\mathbf{G}_*+
 * n\varepsilon_n\mathbf{I})^{-1}\f$) and solve systems \f$\mathbf{GG}_*
 * \mathbf x_i=\mathbf{LL}^\top\mathbf x_i=\mathbf e_i\f$ (\f$\mathbf e_i\f$
 * being the \f$i^{\text{th}}\f$ column of \f$\mathbf I_n\f$) one by one. On
 * the fly it then uses the solution vectors \f$\mathbf x_i\f$ to compute the
 * matrix-matrix product \f$\mathbf C_*=\mathbf G_*\mathbf{GG}_*^{-1}\f$
 * using \f$\mathbf C_{*,(j,i)}=\mathbf G_{*,j}\cdot \mathbf x_i\f$, where
 * $\mathbf G_{*,j}$ is the \f$j^{\text{th}}\f$ row of \f$\mathbf G_*$ (or
 * column, since it is symmetric) and then discarding the vector.
 *
 * The final trace computation is also simplified using the symmetry of the
 * matrices \f$\mathbf R_X\f$ and \f$\mathbf R_Y\f$. Computation of the off-
 * diagonal elements are avoided using
 * \f[
 *	\text{Trace}\left[\mathbf R_X \mathbf R_Y\right ]=\sum_{i=1}^n
 *	\mathbf R_X^i\cdot \mathbf R_Y^i
 * \f]
 *
 * For performing the independence test, ::PERMUTATION test is used by first
 * randomly shuffling the samples from one of the distributions while keeping
 * the samples from the other distribution in the original order. This way we
 * sample the null distribution and compute p-value and threshold for a given
 * test power.
 *
 * [1]: Kenji Fukumizu, Arthur Gretton, Xiaohai Sun, Bernhard Scholkopf:
 * Kernel Measures of Conditional Dependence. NIPS 2007
 */
class CNOCCO : public CKernelIndependenceTest
{
public:
	/** Constructor */
	CNOCCO();

	/** Constructor.
	 *
	 * Initializes the kernels and features from the two distributions and
	 * SG_REFs them
	 *
	 * @param kernel_p kernel to use on samples from p
	 * @param kernel_q kernel to use on samples from q
	 * @param p samples from distribution p
	 * @param q samples from distribution q
	 */
	CNOCCO(CKernel* kernel_p, CKernel* kernel_q, CFeatures* p, CFeatures* q);

	/** Destructor */
	virtual ~CNOCCO();

	/** Computes the NOCCO statistic (see class description) for underlying
	 * kernels and data.
	 *
	 * Note that since kernel matrices have to be stored, it has quadratic
	 * space costs.
	 *
	 * @return unbiased estimate of NOCCO
	 */
	virtual float64_t compute_statistic();

	/** Computes a p-value based on current method for approximating the
	 * null-distribution. The p-value is the 1-p quantile of the null-
	 * distribution where the given statistic lies in.
	 *
	 * @param statistic statistic value to compute the p-value for
	 * @return p-value parameter statistic is the (1-p) percentile of the
	 * null distribution
	 */
	virtual float64_t compute_p_value(float64_t statistic);

	/** Computes a threshold based on current method for approximating the
	 * null-distribution. The threshold is the value that a statistic has
	 * to have in ordner to reject the null-hypothesis.
	 *
	 * @param alpha test level to reject null-hypothesis
	 * @return threshold for statistics to reject null-hypothesis
	 */
	virtual float64_t compute_threshold(float64_t alpha);

	/** @return the class name */
	virtual const char* get_name() const
	{
		return "NOCCO";
	}

	/** @return the statistic type of this test statistic */
	virtual EStatisticType get_statistic_type() const
	{
		return S_NOCCO;
	}

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

	/**
	 * Setter for regularization parameter epsilon
	 * @param epsilon the regularization parameter
	 */
	void set_epsilon(float64_t epsilon);

	/** @return epsilon the regularization parameter */
	float64_t get_epsilon() const;

	/** Merges both sets of samples and computes the test statistic
	 * m_num_null_sample times. This version precomputes the kenrel matrix
	 * once by hand, then samples using this one. The matrix has
	 * to be stored anyway when statistic is computed.
	 *
	 * @return vector of all statistics
	 */
	virtual SGVector<float64_t> sample_null();

protected:
	/**
	 * Helper method which computes the matrix times matrix inverse using LLT
	 * solve (Cholesky) withoout storing the inverse (see class documentation).
	 *
	 * @param m the centered Gram matrix
	 * @return the result matrix of the multiplication
	 */
	SGMatrix<float64_t> compute_helper(SGMatrix<float64_t> m);

private:
	/** Register parameters and initialize with defaults */
	void init();

	/** Number of features from the distributions (should be equal for both) */
	index_t m_num_features;

	/** The regularization constant */
	float64_t m_epsilon;

};

}

#endif // NOCCO_H_
#endif // HAVE_EIGEN3
