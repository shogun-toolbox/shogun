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
class NormOne : public DensePreprocessor<float64_t>
{
	public:
		/** default constructor */
		NormOne();

		/** destructor */
		~NormOne() override;

		/// initialize preprocessor from file
		virtual bool load(FILE* f);
		/// save preprocessor init-data to file
		virtual bool save(FILE* f);

		/// apply preproc on single feature vector
		/// result in feature matrix
		SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector) override;

		/** @return object name */
		const char* get_name() const override { return "NormOne"; }

		/// return a type of preprocessor
		EPreprocessorType get_type() const override { return P_NORMONE; }

	protected:
		SGMatrix<float64_t> apply_to_matrix(SGMatrix<float64_t> matrix) override;
};
}
#endif
