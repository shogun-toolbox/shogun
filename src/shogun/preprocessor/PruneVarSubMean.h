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

#include <shogun/preprocessor/SimplePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/lib/common.h>

#include <stdio.h>

namespace shogun
{
/** @brief Preprocessor PruneVarSubMean will substract the mean and remove
 * features that have zero variance.
 *
 * It will optionally normalize standard deviation of
 * features to 1 (by dividing by standard deviation of the feature)
 */
class CPruneVarSubMean : public CSimplePreprocessor<float64_t>
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
		virtual SGVector<float64_t> apply_to_feature_vector(const SGVector<float64_t>& vector);

		/** @return object name */
		virtual inline const char* get_name() const { return "PruneVarSubMean"; }

		/// return a type of preprocessor
		virtual inline EPreprocessorType get_type() const { return P_PRUNEVARSUBMEAN; }

	protected:
		/** idx */
		int32_t* idx;
		/** mean */
		float64_t* mean;
		/** std */
		float64_t* std;
		/** num idx */
		int32_t num_idx;
		/** divide by std */
		bool divide_by_std;

		/// true when already initialized
		bool initialized;
};
}
#endif
