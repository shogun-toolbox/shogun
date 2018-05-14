/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Yuyu Zhang, Saurabh Goyal, 
 *          Sergey Lisitsyn
 */

#ifndef _CNORM_ONE__H__
#define _CNORM_ONE__H__

#include <shogun/lib/config.h>

#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>


namespace shogun
{
/** @brief Preprocessor NormOne, normalizes vectors to have norm 1.
 *
 * Formally, it computes
 *
 * \f[
 * {\bf x} \leftarrow \frac{{\bf x}}{||{\bf x}||}
 * \f]
 *
 * It therefore does not need any initialization. It is most useful to get data
 * onto a ball of radius one.
 */
class CNormOne : public CDensePreprocessor<float64_t>
{
	public:
		/** default constructor */
		CNormOne();

		/** destructor */
		virtual ~CNormOne();

		/// cleanup
		virtual void cleanup();
		/// initialize preprocessor from file
		virtual bool load(FILE* f);
		/// save preprocessor init-data to file
		virtual bool save(FILE* f);

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

		/** @return object name */
		virtual const char* get_name() const { return "NormOne"; }

		/// return a type of preprocessor
		virtual EPreprocessorType get_type() const { return P_NORMONE; }
};
}
#endif
