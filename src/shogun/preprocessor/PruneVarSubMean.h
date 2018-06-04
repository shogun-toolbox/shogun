/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Yuyu Zhang, Viktor Gal, 
 *          Sergey Lisitsyn, Saurabh Goyal
 */

#ifndef _CPRUNE_VAR_SUB_MEAN__H__
#define _CPRUNE_VAR_SUB_MEAN__H__

#include <shogun/lib/config.h>

#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>


namespace shogun
{
/** @brief Preprocessor PruneVarSubMean will substract the mean and remove
 * features that have zero variance.
 *
 * It will optionally normalize standard deviation of
 * features to 1 (by dividing by standard deviation of the feature)
 */
class CPruneVarSubMean : public CDensePreprocessor<float64_t>
{
	public:
		/** constructor
		 *
		 * @param divide if division shall be made
		 */
		CPruneVarSubMean(bool divide=true);

		/** destructor */
		virtual ~CPruneVarSubMean();

		/// Fit preprocessor into features
		virtual void fit(CFeatures* features);

		/// cleanup
		virtual void cleanup();

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

		/** @return object name */
		virtual const char* get_name() const { return "PruneVarSubMean"; }

		/// return a type of preprocessor
		virtual EPreprocessorType get_type() const { return P_PRUNEVARSUBMEAN; }

	protected:
		virtual SGMatrix<float64_t>
		apply_to_matrix(SGMatrix<float64_t> matrix) override;

	private:
		void init();
		void register_parameters();

	protected:
		/** idx */
		SGVector<int32_t> m_idx;
		/** mean */
		SGVector<float64_t> m_mean;
		/** std */
		SGVector<float64_t> m_std;
		/** num idx */
		int32_t m_num_idx;
		/** divide by std */
		bool m_divide_by_std;

		/// true when already initialized
		bool m_initialized;
};
}
#endif
