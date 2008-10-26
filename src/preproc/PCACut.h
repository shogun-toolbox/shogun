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

#ifndef _CPCACUT__H__
#define _CPCACUT__H__

#include "lib/config.h"

#ifdef HAVE_LAPACK

#include <stdio.h>

#include "preproc/SimplePreProc.h"
#include "features/Features.h"
#include "lib/common.h"


/** Preprocessor PCACut performs principial component analysis on the input
 * vectors and keeps only the n eigenvectors with eigenvalues above a certain
 * threshold. On preprocessing the stored covariance matrix is used to project
 * vectors into eigenspace only returning vectors of reduced dimension n.
 * Optional whitening is performed.
 *
 * This is only useful if the dimensionality of the data is rather low, as the
 * covariance matrix is of size num_feat*num_feat. Note that vectors don't have
 * to have zero mean as it is substracted.
 */
class CPCACut : public CSimplePreProc<DREAL>
{
	public:
		/** constructor
		 *
		 * @param do_whitening do whitening
		 * @param thresh threshold
		 */
		CPCACut(int32_t do_whitening=0, double thresh=1e-6);
		virtual ~CPCACut();

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
		virtual DREAL* apply_to_feature_vector(DREAL* f, int32_t &len);

	protected:
		/** T */
		double* T ;
		/** num dim */
		int32_t num_dim;
		/** num old dim */
		int32_t num_old_dim;
		/** mean */
		double *mean ;

		/// true when already initialized
		bool initialized;

		/** do whitening */
		int32_t do_whitening;
		/** thresh */
		double thresh;
};
#endif
#endif
