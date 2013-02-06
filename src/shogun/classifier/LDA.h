/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LDA_H___
#define _LDA_H___

#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK
#include <shogun/lib/common.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/machine/LinearMachine.h>

namespace shogun
{
	template <class ST> class CDenseFeatures;
/** @brief Class LDA implements regularized Linear Discriminant Analysis.
 *
 * LDA learns a linear classifier and requires examples to be CDenseFeatures.
 * The learned linear classification rule is optimal under the assumption that
 * both classes a gaussian distributed with equal co-variance. To find a linear
 * separation \f${\bf w}\f$ in training, the in-between class variance is
 * maximized and the within class variance is minimized, i.e.
 *
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
		 */
		CLDA(float64_t gamma=0);

		/** constructor
		 *
		 * @param gamma gamma
		 * @param traindat training features
		 * @param trainlab labels for training features
		 */
		CLDA(float64_t gamma, CDenseFeatures<float64_t>* traindat, CLabels* trainlab);
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
		/** gamma */
		float64_t m_gamma;
};
}
#endif
#endif
