/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Michele Mazzoni, 
 *          Evgeniy Andreev, Evan Shelhamer
 */

#ifndef KERNELPCA_H__
#define KERNELPCA_H__
#include <shogun/lib/config.h>

#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/lib/common.h>

namespace shogun
{

class CFeatures;
class CKernel;

/** @brief Preprocessor KernelPCA performs kernel principal component analysis
 *
 * Schoelkopf, B., Smola, A. J., & Mueller, K. R. (1999).
 * Kernel Principal Component Analysis.
 * Advances in kernel methods support vector learning, 1327(3), 327-352. MIT Press.
 * Retrieved from http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.32.8744
 *
 */
class CKernelPCA: public CDimensionReductionPreprocessor
{
public:
		/** default constructor
		 */
		CKernelPCA();

		/** constructor
		 * @param k kernel to be used
		 */
		CKernelPCA(CKernel* k);

		virtual ~CKernelPCA();

		virtual void fit(CFeatures* features);

		/// cleanup
		virtual void cleanup();

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

		/** apply to string features
		 * @param features
		 */
		virtual CDenseFeatures<float64_t>* apply_to_string_features(CFeatures* features);

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

		/** @return object name */
		virtual const char* get_name() const { return "KernelPCA"; }

		/** @return the type of preprocessor */
		virtual EPreprocessorType get_type() const { return P_KERNELPCA; }

	protected:

		/** default init */
		void init();

	protected:

		/** features used by init. needed for apply */
		CFeatures* m_init_features;

		/** transformation matrix */
		SGMatrix<float64_t> m_transformation_matrix;

		/** bias vector */
		SGVector<float64_t> m_bias_vector;

		/** true when already initialized */
		bool m_initialized;

};
}
#endif
