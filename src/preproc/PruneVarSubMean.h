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

class CPruneVarSubMean : public CSimplePreProc<DREAL>
{
	public:
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
		virtual DREAL* apply_to_feature_matrix(CFeatures* f);

		/// apply preproc on single feature vector
		/// result in feature matrix
		virtual DREAL* apply_to_feature_vector(DREAL* f, INT &len);

	protected:
		INT* idx ;
		DREAL* mean;
		DREAL* std;
		INT num_idx ;
		bool divide_by_std;

		/// true when already initialized
		bool initialized;
};
#endif
