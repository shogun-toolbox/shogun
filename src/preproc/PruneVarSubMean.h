/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CPRUNE_VAR_SUB_MEAN__H__
#define _CPRUNE_VAR_SUB_MEAN__H__

#include "preproc/SimplePreProc.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/common.h"

#include <stdio.h>

/** Preprocessor PruneVarSubMean will substract the mean and remove features
 * that have zero variance. It will optionally normalize standard deviation of
 * features to 1 (by dividing by standard deviation of the feature)
 */
class CPruneVarSubMean : public CSimplePreProc<float64_t>
{
	public:
		/** constructor
		 *
		 * @param divide if division shall be made
		 */
		CPruneVarSubMean(bool divide=true);
		virtual ~CPruneVarSubMean();

		/// initialize preprocessor from features
		virtual bool init(CFeatures* f);

		/// initialize preprocessor from file
		virtual bool load_init_data(FILE* src);
		/// save init-data (like transforamtion matrices etc) to file
		virtual bool save_init_data(FILE* dst);

		/// cleanup
		virtual void cleanup();

		/// apply preproc on feature matrix
		/// result in feature matrix
		/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
		virtual float64_t* apply_to_feature_matrix(CFeatures* f);

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual float64_t* apply_to_feature_vector(float64_t* f, int32_t &len);

		/** @return object name */
		inline virtual const char* get_name() { return "PruneVarSubMean"; }

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
#endif
