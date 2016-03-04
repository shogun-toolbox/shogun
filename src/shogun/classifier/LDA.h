/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2014 Abhijeet Kislay
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LDA_H___
#define _LDA_H___

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/machine/LinearMachine.h>

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
	/** Fisher two class discrimiant based LDA.
	*/
	FLD_LDA = 30
};

template <class ST> class CDenseFeatures;
/** @brief Class LDA implements regularized Linear Discriminant Analysis.
 *
 * LDA learns a linear classifier and requires examples to be CDenseFeatures.
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
 * \f[S_w := \sum_{c\in\{-1,+1\}}\sum_{{\bf x}\in X_{c}}({\bf x} - {\bf m_c})({\bf x} - {\bf m_c})^T \f]
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
 * class matrix. However, this condition gets void when there are fewer data-points
 * than dimensions. A solution is to require that \f${\bf W}\f$ lies only in the subspace
 * spanned by the data. A basis of the data \f${\bf X}\f$ is found using the thin-SVD technique
 * which returns an orthonormal non-square basis matrix \f${\bf Q}\f$. We then require the
 * solution \f${\bf w}\f$ to be expressed in this basis.
 * \f[{\bf W} := {\bf Q} {\bf{W^\prime}}\f]
 * The between class Matrix is replaced with:
 * \f[{\bf S_b^\prime} \equiv {{\bf Q^T}{\bf S_b } {\bf Q}\f]
 * The within class Matrix is replaced with:
 * \f[{\bf S_w^\prime} \equiv {{\bf Q^T}{\bf S_w} {\bf Q}\f]
 * In this case {\bf S_w^\prime} is guranteed invertible since {\bf S_w} has
 * been projected down to the basis that spans the data.
 * see: Bayesian Reasoning and Machine Learning, section 16.3.1.
 *
 * <em>::AUTO_LDA</em> : This mode automagically chooses one of the above modes for
 * the users based on whether N > D (chooses ::FLD_LDA) or N < D(chooses ::SVD_LDA)
 * \sa CLinearMachine
 * \sa http://en.wikipedia.org/wiki/Linear_discriminant_analysis
 */

class CLDA : public CLinearMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_BINARY);

		/** constructor
		 *
		 * @param gamma gamma
		 * @param method LDA using Fisher's algorithm or Singular Value Decomposition : ::FLD_LDA/::SVD_LDA/::AUTO_LDA[default]
		 */
		CLDA(float64_t gamma=0, ELDAMethod method=AUTO_LDA);

		/** constructor
		 *
		 * @param gamma gamma
		 * @param traindat training features
		 * @param trainlab labels for training features
		 * @param method LDA using Fisher's algorithm or Singular Value Decomposition : ::FLD_LDA/::SVD_LDA/::AUTO_LDA[default]

		 */
		CLDA(float64_t gamma, CDenseFeatures<float64_t>* traindat, CLabels* trainlab, ELDAMethod method=AUTO_LDA);
		virtual ~CLDA();

		/** set gamma
		 *
		 * @param gamma the new gamma
		 */
		inline void set_gamma(float64_t gamma)
		{
			m_gamma=gamma;
		}

		/** get gamma
		 *
		 * @return gamma
		 */
		inline float64_t get_gamma()
		{
			return m_gamma;
		}

		/** get classifier type
		 *
		 * @return classifier type LDA
		 */
		virtual EMachineType get_classifier_type()
		{
			return CT_LDA;
		}

		/** set features
		 *
		 * @param feat features to set
		 */
		virtual void set_features(CDotFeatures* feat)
		{
			if (feat->get_feature_class() != C_DENSE ||
				feat->get_feature_type() != F_DREAL)
				SG_ERROR("LDA requires SIMPLE REAL valued features\n")

			CLinearMachine::set_features(feat);
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
		virtual bool train_machine(CFeatures* data=NULL);

	protected:

		void init();

		/** gamma */
		float64_t m_gamma;
		/** LDA mode */
		ELDAMethod m_method;
};
}
#endif//ifndef
