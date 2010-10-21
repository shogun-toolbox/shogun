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

#ifndef _CPCACUT__H__
#define _CPCACUT__H__

#include "lib/config.h"

#ifdef HAVE_LAPACK

#include <stdio.h>

#include "preproc/SimplePreProc.h"
#include "features/Features.h"
#include "lib/common.h"

namespace shogun
{
/** @brief Preprocessor PCACut performs principial component analysis on the input
 * vectors and keeps only the n eigenvectors with eigenvalues above a certain
 * threshold.
 *
 * On preprocessing the stored covariance matrix is used to project
 * vectors into eigenspace only returning vectors of reduced dimension n.
 * Optional whitening is performed.
 *
 * This is only useful if the dimensionality of the data is rather low, as the
 * covariance matrix is of size num_feat*num_feat. Note that vectors don't have
 * to have zero mean as it is substracted.
 */
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class CPCACut : public CSimplePreProc<float64_t>
{
	public:
		/** constructor
		 *
		 * @param do_whitening do whitening
		 * @param thresh threshold
		 */
		CPCACut(int32_t do_whitening=0, float64_t thresh=1e-6);
		virtual ~CPCACut();

		/// initialize preprocessor from features
		virtual bool init(CFeatures* f);
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
		inline virtual const char* get_name() { return "PCACut"; }

	protected:
		/** T */
		double* T ;
		/** num dim */
		int32_t num_dim;
		/** num old dim */
		int32_t num_old_dim;
		/** mean */
		float64_t *mean ;

		/// true when already initialized
		bool initialized;

		/** do whitening */
		int32_t do_whitening;
		/** thresh */
		float64_t thresh;
};
}
#endif
#endif
