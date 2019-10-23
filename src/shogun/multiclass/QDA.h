/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Michele Mazzoni, Kevin Hughes,
 *          Thoralf Klein, Fernando Iglesias, Sergey Lisitsyn, Pan Deng
 */

#ifndef _QDA_H__
#define _QDA_H__

#include <shogun/lib/config.h>


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
class QDA : public NativeMulticlassMachine
{
	public:
		MACHINE_PROBLEM_TYPE(PT_MULTICLASS)

		/** default constructor */
		QDA();

		/** constructor
		 *
		 * @param tolerance tolerance used in training
		 * @param store_covs whether to store the within class covariances
		 */
		QDA(float64_t tolerance, bool store_covs);

		/** constructor
		 *
		 * @param traindat training features
		 * @param trainlab labels for training features
		 */
		QDA(const std::shared_ptr<DenseFeatures<float64_t>>& traindat, std::shared_ptr<Labels> trainlab);

		/** constructor
		 *
		 * @param traindat training features
		 * @param trainlab labels for training features
		 * @param tolerance tolerance used in training
		 */
		QDA(const std::shared_ptr<DenseFeatures<float64_t>>& traindat, std::shared_ptr<Labels> trainlab, float64_t tolerance);

		/** constructor
		 *
		 * @param traindat training features
		 * @param trainlab labels for training features
		 * @param store_covs whether to store the within class covariances
		 */
		QDA(const std::shared_ptr<DenseFeatures<float64_t>>& traindat, std::shared_ptr<Labels> trainlab, bool store_covs);

		/** constructor
		 *
		 * @param traindat training features
		 * @param trainlab labels for training features
		 * @param tolerance tolerance used in training
		 * @param store_covs whether to store the within class covariances
		 */
		QDA(const std::shared_ptr<DenseFeatures<float64_t>>& traindat, std::shared_ptr<Labels> trainlab, float64_t tolerance, bool store_covs);

		virtual ~QDA();

		/** apply QDA to data
		 *
		 * @param data (test) data to be classified
		 * @return labels result of classification
		 */
		virtual std::shared_ptr<MulticlassLabels> apply_multiclass(std::shared_ptr<Features> data=NULL);

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
		virtual void set_features(std::shared_ptr<DotFeatures> feat)
		{
			if (feat->get_feature_class() != C_DENSE ||
				feat->get_feature_type() != F_DREAL)
				error("QDA requires SIMPLE REAL valued features");



			m_features = feat;
		}

		/** get features
		 *
		 * @return features
		 */
		virtual std::shared_ptr<DotFeatures> get_features() {  return m_features; }

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
			require(m_store_covs, "Covariance matrices were not stored. "
					"Please activate to access them subsequently.");
			return SGMatrix< float64_t >(m_covs.get_matrix(c), m_dim, m_dim, false);
		}

	protected:
		/** train QDA classifier
		 *
		 * @param data training data
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(std::shared_ptr<Features> data = NULL);

	private:
		void init();

		void cleanup();

	private:
		/** feature vectors */
		std::shared_ptr<DotFeatures> m_features;

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

		/** matrices computed in training and used in classification
		 * the matrices are stacked horizontally into a matrix of size
		 * (m_dim, m_dim*m_num_classes).
		 */
		SGMatrix<float64_t> m_M;

		/** vector computed in training and used in classification */
		SGVector< float32_t > m_slog;

}; /* class QDA */
}  /* namespace shogun */

#endif /* _QDA_H__ */
