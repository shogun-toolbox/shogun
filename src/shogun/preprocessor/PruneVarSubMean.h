/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
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

		/// initialize preprocessor from features
		virtual bool init(CFeatures* features);

		/// cleanup
		virtual void cleanup();

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

		/** @return object name */
		virtual const char* get_name() const { return "PruneVarSubMean"; }

		/// return a type of preprocessor
		virtual EPreprocessorType get_type() const { return P_PRUNEVARSUBMEAN; }

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
