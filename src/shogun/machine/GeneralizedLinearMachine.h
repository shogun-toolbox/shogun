/*
 * STILL A WORK IN PROGRESS
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Tej Sukhatme
 */

// #ifndef _LINEARCLASSIFIER_H__
// #define _LINEARCLASSIFIER_H__

// #include <shogun/lib/config.h>

// #include <shogun/lib/common.h>
#include <shogun/machine/Machine.h>
// #include <shogun/lib/SGVector.h>


namespace shogun
{

// class BinaryLabels;
// class DotFeatures;
// class Features;
// class RegressionLabels;

/** @brief Class for estimating regularized generalized linear models (GLM).
 *   The regularized GLM minimizes the penalized negative log likelihood:
 *      .. math::
 *
 *		\\min_{\\beta_0, \\beta} \\frac{1}{N}
 *		\\sum_{i = 1}^N \\mathcal{L} (y_i, \\beta_0 + \\beta^T x_i)
 *		+ \\lambda [ \\frac{1}{2}(1 - \\alpha) \\mathcal{P}_2 +
 *						\\alpha \\mathcal{P}_1 ]
 *
 *	where :math:`\\mathcal{P}_2` and :math:`\\mathcal{P}_1` are the generalized
 *	L2 (Tikhonov) and generalized L1 (Group Lasso) penalties, given by:
 *
 *	.. math::
 *
 *		\\mathcal{P}_2 = \\|\\Gamma \\beta \\|_2^2 \\
 *		\\mathcal{P}_1 = \\sum_g \\|\\beta_{j,g}\\|_2
 *
 *	where :math:`\\Gamma` is the Tikhonov matrix: a square factorization
 *	of the inverse covariance matrix and :math:`\\beta_{j,g}` is the
 *	:math:`j` th coefficient of group :math:`g`.
 *
 *	The generalized L2 penalty defaults to the ridge penalty when
 *	:math:`\\Gamma` is identity.
 *
 *	The generalized L1 penalty defaults to the lasso penalty when each
 *	:math:`\\beta` belongs to its own group.
 *
 * */
class GeneralizedLinearMachine : public Machine
{
// 	public:
// 		/** default constructor */
// 		LinearMachine();

// 		/** destructor */
// 		virtual ~LinearMachine();

// 		/** copy constructor */
// 		LinearMachine(const std::shared_ptr<LinearMachine>& machine);

// 		/** get w
// 		 *
// 		 * @return weight vector
// 		 */
// 		virtual SGVector<float64_t> get_w() const;

// 		/** set w
// 		 *
// 		 * @param src_w new w
// 		 */
// 		virtual void set_w(const SGVector<float64_t> src_w);

// 		/** set bias
// 		 *
// 		 * @param b new bias
// 		 */
// 		virtual void set_bias(float64_t b);

// 		/** get bias
// 		 *
// 		 * @return bias
// 		 */
// 		virtual float64_t get_bias() const;

// 		/** set features
// 		 *
// 		 * @param feat features to set
// 		 */
// 		virtual void set_features(std::shared_ptr<DotFeatures> feat);

// 		/** apply linear machine to data
// 		 * for binary classification problem
// 		 *
// 		 * @param data (test)data to be classified
// 		 * @return classified labels
// 		 */
// 		virtual std::shared_ptr<BinaryLabels> apply_binary(std::shared_ptr<Features> data=NULL);

// 		/** apply linear machine to data
// 		 * for regression problem
// 		 *
// 		 * @param data (test)data to be classified
// 		 * @return classified labels
// 		 */
// 		virtual std::shared_ptr<RegressionLabels> apply_regression(std::shared_ptr<Features> data=NULL);

// 		/** applies to one vector */
// 		virtual float64_t apply_one(int32_t vec_idx);

// 		/** get features
// 		 *
// 		 * @return features
// 		 */
// 		virtual std::shared_ptr<DotFeatures> get_features();

// 		/** Returns the name of the SGSerializable instance.  It MUST BE
// 		 *  the CLASS NAME without the prefixed `C'.
// 		 *
// 		 * @return name of the SGSerializable
// 		 */
// 		virtual const char* get_name() const { return "LinearMachine"; }

// 	protected:

// 		/** apply get outputs
// 		 *
// 		 * @param data features to compute outputs
// 		 * @return outputs
// 		 */
// 		virtual SGVector<float64_t> apply_get_outputs(std::shared_ptr<Features> data);

// 	private:

// 		void init();

// 	protected:
// 		/** w */
// 		SGVector<float64_t> m_w;

// 		/** bias */
// 		float64_t bias;

// 		/** features */
// 		std::shared_ptr<DotFeatures> features;
};
}
// #endif
