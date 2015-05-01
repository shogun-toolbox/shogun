/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2014 Abhijeet Kislay
 */

#ifndef LDA_H_
#define LDA_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/preprocessor/Preprocessor.h>
#include <shogun/labels/Labels.h>
#include <shogun/features/Features.h>

namespace shogun
{

	/** Matrix decomposition method for Fisher LDA */
	enum EFLDAMethod
	{
		/** if N>D then ::CLASSIC_FLDA is chosen automatically else ::CANVAR_FLDA is chosen
		 * (D-dimensions N-number of vectors)
		 */
		AUTO_FLDA = 10,
		/** Cannonical Variable based FLDA. */
		CANVAR_FLDA = 20,
		/** Classical Fisher Linear Discriminant Analysis. */
		CLASSIC_FLDA = 30
	};

	/** @brief Preprocessor FisherLDA attempts to model the difference between the classes
	 * of data by performing linear discriminant analysis on input feature vectors/matrices.
	 * When the init method in FisherLDA is called with proper feature matrix X(say N number
	 * of vectors and D feature dimensions) supplied via apply_to_feature_matrix or
	 * apply_to_feature_vector methods, this creates a transformation whose outputs are the
	 * reduced T-Dimensional & class-specific distribution (where T<= number of unique
	 * classes-1). The transformation matrix is essentially a DxT matrix, the columns of
	 * which correspond to the specified number of eigenvectors which maximizes the ratio
	 * of between class matrix to within class matrix.
	 *
	 * This class provides 3 method options to compute the transformation matrix :
	 *
	 * <em>::CLASSIC_FLDA</em> : This method selects W in such a way that the ratio of the
	 * between-class scatter and the within class scatter is maximized.
	 * The between class matrix is :
	 * \f$\sum_b = \sum_{i=1}^C{\bf{(\mu_i-\mu)(\mu_i-\mu)^T}}\f$
	 * The within class matrix is :
	 * \f$\sum_w = \sum_{i=1}^C{\sum_{x_k\in}^c{\bf{(\mu_i-\mu)(\mu_i-\mu)^T}}}\f$
	 * This should be choosen when N>D
	 *
	 * <em>::CANVAR_FLDA</em> : This method performs Canonical Variates which
	 * generalises Fisher's method to projection of more than one dimension.
	 * This is equipped to handle the cases where the within class matrix
	 * are non-invertible. Can be used for both cases(D>N or D<N). See the
	 * implementation in Bayesian Reasoning and Machine Learning by David Barber
	 * , Section 16.3
	 *
	 *
	 * <em>::AUTO_FLDA</em> : Automagically, the appropriate method is selected based on
	 * whether D>N (chooses ::CANVAR_FLDA) or D<N(chooses ::CLASSIC_FLDA)
	 */
class CFisherLDA: public CDimensionReductionPreprocessor
{
	public:

		/** standard constructor
		 * @param method LDA based on : ::CLASSIC_FLDA/::CANVAR_FLDA/::AUTO_FLDA[default]
		 * @param thresh threshold value for ::CANVAR_FLDA only. This is used to reject
		 * those basis whose singular values are less than the provided threshold.
		 * The default one is 0.01.
		 */
		CFisherLDA(EFLDAMethod method=AUTO_FLDA, float64_t thresh=0.01);

		/** destructor */
		virtual ~CFisherLDA();

		/** fits fisher lda transformation using features and corresponding labels
		 * @param features using which the transformation matrix will be formed
		 * @param labels of the given features which will be used here to find
		 * the transformation matrix unlike PCA where it is not needed.
		 * @param dimensions number of dimensions to retain
		 */
		virtual bool fit(CFeatures* features, CLabels* labels, int32_t num_dimensions=0);

		/** cleanup */
		virtual void cleanup();

		/** apply preprocessor to feature matrix
		 * @param features on which the learned tranformation has to be applied.
		 * Sometimes it is also referred as projecting the given features.
		 * @return processed feature matrix with reduced dimensions.
		 */
		virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

		/** apply preprocessor to feature vector
		 * @param features on which the learned transformation has to be applied.
		 * @return processed feature vector with reduced dimensions.
		 */
		virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

		/** @return get transformation matrix which contains the required number of eigenvectors
		*/
		SGMatrix<float64_t> get_transformation_matrix();

		/** @return get eigenvalues of LDA
		*/
		SGVector<float64_t> get_eigenvalues();

		/** @return get mean vector of the original data
		*/
		SGVector<float64_t> get_mean();

		/** @return object name */
		virtual const char* get_name() const {return "FisherLDA";}

		/** @return a type of preprocessor */
		virtual EPreprocessorType get_type() const {return P_FISHERLDA;}

	private:

		void initialize();

	protected:


		/** transformation matrix */
		SGMatrix<float64_t> m_transformation_matrix;
		/** num dim */
		int32_t m_num_dim;
		/** m_threshold */
		float64_t m_threshold;
		/** m_method */
		int32_t m_method;
		/** mean vector */
		SGVector<float64_t> m_mean_vector;
		/** eigenvalues vector */
		SGVector<float64_t> m_eigenvalues_vector;
};
}
#endif //HAVE_EIGEN3
#endif //ifndef
