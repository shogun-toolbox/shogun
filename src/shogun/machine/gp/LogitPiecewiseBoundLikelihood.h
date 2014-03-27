/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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
 *
 */

#ifndef _LOGITPIECEWISEBOUNDLIKELIHOOD_H_
#define _LOGITPIECEWISEBOUNDLIKELIHOOD_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/BinaryLabels.h>

namespace shogun
{



/** @brief Class that models Logit likelihood with variational piecewise bound
 *
 */
class CLogitPiecewiseBoundLikelihood : public CLogitLikelihood
{
typedef enum _my_bsxfunOp
{
	plus,
	times
} MyBsxfunOp;
public:
	CLogitPiecewiseBoundLikelihood();

	virtual ~CLogitPiecewiseBoundLikelihood();

	/** returns the name of the likelihood model
	 *
	 * @return name LogitPiecewiseBoundLikelihood
	 */
	virtual const char* get_name() const { return "LogitPiecewiseBoundLikelihood"; }




	/** set the variational piecewise bound for logit likelihood
	 *
	 *  @param bound variational piecewise bound
	 */
	virtual void set_bound(SGMatrix<float64_t> bound);


	/** set the variational normal distribution given data and parameters
	 *
	 * @param mu mean of the variational normal distribution
	 * @param s2 variance of the variational normal distribution
	 * @param lab labels/data used
	 *
	 */
	virtual void set_distribution(SGVector<float64_t> mu, SGVector<float64_t> s2, const CLabels* lab);

	/** returns the expection of the logarithm of a logit distribution 
	 * wrt the variational distribution using piecewise bound
	 *
	 * @return expection
	 */
	virtual SGVector<float64_t> get_variational_expection();

	/** get derivative of the variational expection of log LogitLikelihood
	 * using the piecewise bound with respect to given parameter
	 *
	 * @param param parameter
	 *
	 * @return derivative
	 */
	virtual SGVector<float64_t> get_variational_first_derivative(const TParameter* param) const;
private:
	void init();

	void clean();

	void precompute();

	void init_matrix(SGMatrix<float64_t> ** ptr, index_t num_rows, index_t num_cols);

	SGMatrix<float64_t> * m_bound;

	SGVector<float64_t> * m_mu;

	SGVector<float64_t> * m_s2;

	SGVector<float64_t> * m_lab;

	//rows = l.size   cols = m.size
	SGMatrix<float64_t> * m_pl;

	SGMatrix<float64_t> * m_ph;

	SGMatrix<float64_t> * m_cdf_diff;

	SGMatrix<float64_t> * m_l2_plus_s2;

	SGMatrix<float64_t> * m_h2_plus_s2;

	SGMatrix<float64_t> * m_weighted_pdf_diff;


	// It seems that pl8787 tries to include it into the cade base
	// Temporarily add here for unit test
	static Eigen::MatrixXd my_bsxfun_vec(MyBsxfunOp op, const Eigen::MatrixXd & x,
		const Eigen::VectorXd & y, bool is_col_vec);

	template<typename M1, typename M2>
	static Eigen::MatrixXd my_bsxfun(MyBsxfunOp op, const Eigen::MatrixBase<M1> & x,
		const Eigen::MatrixBase<M2> & y);

	static float64_t _standard_norm_pdf(float64_t x);

	template<typename M1>
	static Eigen::MatrixXd standard_norm_pdf(const Eigen::MatrixBase<M1> &x);

	static float64_t _norm_cdf_minus_const(float64_t x);

	template<typename M1>
	static Eigen::MatrixXd normal_cdf_minus_const(const Eigen::MatrixBase<M1> &x);

	static float64_t _check_variance(float64_t x);
	static float64_t _convert_label(float64_t x);
	// end for "temporarily add for unit test"
};
}
#endif /* HAVE_EIGEN3 */
#endif /* _LOGITPIECEWISEBOUNDLIKELIHOOD_H_ */
