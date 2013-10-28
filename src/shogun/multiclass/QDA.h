/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _QDA_H__
#define _QDA_H__

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/features/DotFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/machine/NativeMulticlassMachine.h>
#include <shogun/lib/SGNDArray.h>

namespace shogun
{

//#define DEBUG_QDA

/** @brief Class QDA implements Quadratic Discriminant Analysis.
 *
 *  QDA learns a quadratic classifier and requires examples to be CSimplefeatures.
 *  This classifier is optimal under the assumptions that the classes are normally
 *  distributed (i.e. they are Gaussians). Unlike LDA, QDA does not assume that all
 *  the classes are distributed with equal co-variance.
 *  TODO
 */
class CQDA : public CNativeMulticlassMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_MULTICLASS)

		/** constructor
		 *
		 * @param tolerance tolerance used in training
		 * @param store_covs whether to store the within class covariances
		 */
		CQDA(float64_t tolerance = 1e-4, bool store_covs = false);

		/** constructor
		 *
		 * @param traindat training features
		 * @param trainlab labels for training features
		 * @param tolerance tolerance used in training
		 * @param store_covs whether to store the within class covariances
		 */
		CQDA(CDenseFeatures<float64_t>* traindat, CLabels* trainlab, float64_t tolerance = 1e-4, bool store_covs = false);

		virtual ~CQDA();

		/** apply QDA to data
		 *
		 * @param data (test) data to be classified
		 * @return labels result of classification
		 */
		virtual CMulticlassLabels* apply_multiclass(CFeatures* data=NULL);

		/** set store_covs
		 *
		 * @param store_covs whether to store the within class covariances
		 */
		inline void set_store_covs(bool store_covs) { m_store_covs = store_covs; }

		/** get store_covs
		 *
		 * @return store_covs whether the class covariances are stored
		 */
		inline bool get_store_covs() { return m_store_covs; }

		/** set tolerance
		 *
		 * @param tolerance tolerance used during training
		 */
		inline void set_tolerance(float64_t tolerance) { m_tolerance = tolerance; }

		/** get tolerance
		 *
		 * @return tolerance used during training
		 */
		inline bool get_tolerance() { return m_tolerance; }

		/** get classifier type
		 *
		 * @return classifier type QDA
		 */
		virtual EMachineType get_classifier_type() { return CT_QDA; }

		/** set features
		 *
		 * @param feat features to set
		 */
		virtual void set_features(CDotFeatures* feat)
		{
			if (feat->get_feature_class() != C_DENSE ||
				feat->get_feature_type() != F_DREAL)
				SG_ERROR("QDA requires SIMPLE REAL valued features\n")

			SG_REF(feat);
			SG_UNREF(m_features);
			m_features = feat;
		}

		/** get features
		 *
		 * @return features
		 */
		virtual CDotFeatures* get_features() { SG_REF(m_features); return m_features; }

		/** get object name
		 *
		 * @return object name
		 */
		virtual const char* get_name() const { return "QDA"; }

		/** get a class' mean vector
		 *
		 * @param c class index
		 *
		 * @return mean vector of class c
		 */
		inline SGVector< float64_t > get_mean(int32_t c) const
		{
			return SGVector< float64_t >(m_means.get_column_vector(c), m_dim, false);
		}

		/** get a class' covariance matrix
		 *
		 * @param c class index
		 *
		 * @return covariance matrix of class c
		 */
		inline SGMatrix< float64_t > get_cov(int32_t c) const
		{
			return SGMatrix< float64_t >(m_covs.get_matrix(c), m_dim, m_dim, false);
		}

	protected:
		/** train QDA classifier
		 *
		 * @param data training data
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data = NULL);

	private:
		void init();

		void cleanup();

	private:
		/** feature vectors */
		CDotFeatures* m_features;

		/** tolerance used during training */
		float64_t m_tolerance;

		/** whether to store the within class covariances */
		bool m_store_covs;

		/** number of classes */
		int32_t m_num_classes;

		/** dimension of the features space */
		int32_t m_dim;

		/** feature covariances for each of the classes in the training data
		 *  stored iif store_covs
		 */
		SGNDArray< float64_t > m_covs;

		/** feature means for each of the classes in the training data */
		SGMatrix< float64_t > m_means;

		/** matrices computed in training and used in classification */
		SGNDArray< float64_t > m_M;

		/** vector computed in training and used in classification */
		SGVector< float32_t > m_slog;

}; /* class QDA */
}  /* namespace shogun */

#endif /* HAVE_EIGEN3 */
#endif /* _QDA_H__ */
