/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Michele Mazzoni,
 *          Evgeniy Andreev, Evan Shelhamer
 */

#ifndef KERNELPCA_H__
#define KERNELPCA_H__
#include <shogun/lib/config.h>

#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/lib/common.h>
#include <shogun/preprocessor/DensePreprocessor.h>

namespace shogun
{

class Features;
class Kernel;

/** @brief Preprocessor KernelPCA performs kernel principal component analysis
 *
 * Schoelkopf, B., Smola, A. J., & Mueller, K. R. (1999).
 * Kernel Principal Component Analysis.
 * Advances in kernel methods support vector learning, 1327(3), 327-352. MIT Press.
 * Retrieved from http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.32.8744
 *
 */
class KernelPCA : public Preprocessor
{
public:
		/** default constructor
		 */
		KernelPCA();

		/** constructor
		 * @param k kernel to be used
		 */
		KernelPCA(std::shared_ptr<Kernel> k);

		virtual ~KernelPCA();

		virtual void fit(std::shared_ptr<Features> features);

		/** Apply transformation to features. In-place mode is not supported.
		 *	@param features features to transform
		 *	@param inplace whether transform in place
		 *	@return the result feature object after applying the transformer
		 */
		virtual std::shared_ptr<Features> transform(std::shared_ptr<Features> features, bool inplace = true);

		/// cleanup
		virtual void cleanup();

		virtual SGMatrix<float64_t> apply_to_feature_matrix(std::shared_ptr<Features> features);

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

		/** apply to string features
		 * @param features
		 */
		virtual std::shared_ptr<DenseFeatures<float64_t>> apply_to_string_features(std::shared_ptr<Features> features);

		/** get transformation matrix, i.e. eigenvectors
		 *
		 */
		SGMatrix<float64_t> get_transformation_matrix() const
		{
			return m_transformation_matrix;
		}

		/** get bias of KPCA
		 *
		 */
		SGVector<float64_t> get_bias_vector() const
		{
			return m_bias_vector;
		}

		virtual EFeatureClass get_feature_class();

		virtual EFeatureType get_feature_type();

		/** @return object name */
		virtual const char* get_name() const { return "KernelPCA"; }

		/** @return the type of preprocessor */
		virtual EPreprocessorType get_type() const { return P_KERNELPCA; }

		/** setter for target dimension
		 * @param dim target dimension
		 */
		void set_target_dim(int32_t dim);

		/** getter for target dimension
		 * @return target dimension
		 */
		int32_t get_target_dim() const;

		/** setter for kernel
		 * @param kernel kernel to set
		 */
		void set_kernel(std::shared_ptr<Kernel> kernel);

		/** getter for kernel
		 * @return kernel
		 */
		std::shared_ptr<Kernel> get_kernel() const;

	protected:

		/** default init */
		void init();

	protected:

		/** features used by init. needed for apply */
		std::shared_ptr<Features> m_init_features;

		/** transformation matrix */
		SGMatrix<float64_t> m_transformation_matrix;

		/** bias vector */
		SGVector<float64_t> m_bias_vector;

		/** target dimension */
		int32_t m_target_dim;

		/** kernel to be used */
		std::shared_ptr<Kernel> m_kernel;
};
}
#endif
