/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Michele Mazzoni, Heiko Strathmann,
 *          Fernando Iglesias, Sergey Lisitsyn, Abhijeet Kislay, Bjoern Esser,
 *          Christopher Goldsworthy, Sanuj Sharma
 */

#ifndef _LDA_H___
#define _LDA_H___

#include <shogun/lib/config.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>
#include <shogun/machine/FeatureDispatchCRTP.h>

namespace shogun
{

/** Method for solving LDA */
enum ELDAMethod
{
	/** if N>D then FLD_LDA is chosen automatically else SVD_LDA is chosen
	 * (D-dimensions N-number of vectors)
	 */
	AUTO_LDA = 10,
	/** Singular Value Decomposition based LDA.
	*/
	SVD_LDA = 20,
	/** Fisher two class discriminant based LDA.
	*/
	FLD_LDA = 30
};

template <class ST> class DenseFeatures;
/** @brief Class LDA implements regularized Linear Discriminant Analysis.
 *
 * LDA learns a linear classifier and requires examples to be DenseFeatures.
 * The learned linear classification rule is optimal under the assumption that
 * both classes a gaussian distributed with equal co-variance. To find a linear
 * separation \f${\bf w}\f$ in training, the in-between class variance is
 * maximized and the within class variance is minimized.
 *
 * This class provides 3 method options to compute the LDA :
 * <em>::FLD_LDA</em> : Two class Fisher Discriminant Analysis.
 * \f[
 * J({\bf w})=\frac{{\bf w^T} S_B {\bf w}}{{\bf w^T} S_W {\bf w}}
 * \f]
 *
 * is maximized, where
 * \f[S_b := ({\bf m_{+1}} - {\bf m_{-1}})({\bf m_{+1}} - {\bf m_{-1}})^T \f]
 * is the between class scatter matrix and
 * \f[S_w := \sum_{c\in\{-1,+1\}}\sum_{{\bf x}\in X_{c}}({\bf x} - {\bf
 * m_c})({\bf x} - {\bf m_c})^T \f]
 * is the within class scatter matrix with mean \f${\bf m_c} :=
 * \frac{1}{N}\sum_{j=1}^N {\bf x_j^c}\f$ and \f$X_c:=\{x_1^c, \dots, x_N^c\}\f$
 * the set of examples of class c.
 *
 * LDA is very fast for low-dimensional samples. The regularization parameter
 * \f$\gamma\f$ (especially useful in the low sample case) should be tuned in
 * cross-validation.
 *
 * <em>::SVD_LDA</em> : Singular Valued decomposition method.
 * The above derivation of Fisher's LDA requires the invertibility of the within
 * class matrix. However, this condition gets void when there are fewer
 * data-points
 * than dimensions. A solution is to require that \f${\bf W}\f$ lies only in the
 * subspace
 * spanned by the data. A basis of the data \f${\bf X}\f$ is found using the
 * thin-SVD technique
 * which returns an orthonormal non-square basis matrix \f${\bf Q}\f$. We then
 * require the
 * solution \f${\bf w}\f$ to be expressed in this basis.
 * \f[{\bf W} := {\bf Q} {\bf{W^\prime}}\f]
 * The between class Matrix is replaced with:
 * \f[{\bf S_b^\prime} \equiv {\bf Q^T}{\bf S_b}{\bf Q}\f]
 * The within class Matrix is replaced with:
 * \f[{\bf S_w^\prime} \equiv {\bf Q^T}{\bf S_w}{\bf Q}\f]
 * In this case {\bf S_w^\prime} is guranteed invertible since {\bf S_w} has
 * been projected down to the basis that spans the data.
 * see: Bayesian Reasoning and Machine Learning, section 16.3.1.
 *
 * <em>::AUTO_LDA</em> : This mode automagically chooses one of the above modes
 * for
 * the users based on whether N > D (chooses ::FLD_LDA) or N < D(chooses
 * ::SVD_LDA)
 * Note that even if N > D FLD_LDA may fail being the covariance matrix not
 * invertible,
 * in such case one should use SVD_LDA.
 * \sa LinearMachine
 * \sa http://en.wikipedia.org/wiki/Linear_discriminant_analysis
 */

class LDA : public DenseRealDispatch<LDA, LinearMachine>
{
	friend class DenseRealDispatch<LDA, LinearMachine>;
	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		/** constructor
		 *
		 * @param gamma gamma
		 * @param method LDA using Fisher's algorithm or Singular Value
		 * Decomposition : ::FLD_LDA/::SVD_LDA/::AUTO_LDA[default]
		 * @param bdc_svd when using SVD solver switch between
		 * Bidiagonal Divide and Conquer algorithm (BDC) and
		 * Jacobi's algorithm, for the differences @see linalg::SVDAlgorithm.
		 * [default = BDC-SVD]
		 */
		LDA(
		    float64_t gamma = 0, ELDAMethod method = AUTO_LDA,
		    bool bdc_svd = true);

		/** constructor
		 *
		 * @param gamma gamma
		 * @param traindat training features
		 * @param trainlab labels for training features
		 * @param method LDA using Fisher's algorithm or Singular Value
		 * Decomposition : ::FLD_LDA/::SVD_LDA/::AUTO_LDA[default]
		 * @param bdc_svd when using SVD solver switch between
		 * Bidiagonal Divide and Conquer algorithm (BDC-SVD) and
		 * Jacobi's algorithm, for the differences @see linalg::SVDAlgorithm.
		 * [default = BDC-SVD]
		 */
		LDA(
		    float64_t gamma, const std::shared_ptr<DenseFeatures<float64_t>>& traindat,
		    std::shared_ptr<Labels> trainlab, ELDAMethod method = AUTO_LDA,
		    bool bdc_svd = true);
		virtual ~LDA();

		/** get classifier type
		 *
		 * @return classifier type LDA
		 */
		virtual EMachineType get_classifier_type()
		{
			return CT_LDA;
		}

		/** @return object name */
		virtual const char* get_name() const { return "LDA"; }

	protected:
		/** train LDA classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		template <typename ST, typename U = typename std::enable_if_t<
		                           std::is_floating_point<ST>::value>>
		bool train_machine_templated(std::shared_ptr<DenseFeatures<ST>> data);

		/**
		 * Train the machine with the svd-based solver (@see CFisherLDA).
		 * @param features training data
		 * @param labels labels for training data
		 */
		template <typename ST>
		bool solver_svd(std::shared_ptr<DenseFeatures<ST>> data);

		/**
		 * Train the machine with the classic method based on the cholesky
		 * decomposition of the covariance matrix.
		 * @param features training data
		 * @param labels labels for training data
		 */
		template <typename ST>
		bool solver_classic(std::shared_ptr<DenseFeatures<ST>> data);

	protected:

		void init();

		/** gamma */
		float64_t m_gamma;
		/** LDA mode */
		ELDAMethod m_method;
		/** use bdc-svd algorithm */
		bool m_bdc_svd;
};
}
#endif//ifndef
