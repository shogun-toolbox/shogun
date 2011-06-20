/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 1999-2008,2011 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2011 Berlin Institute of Technology
 */

#ifndef _CPCACUT__H__
#define _CPCACUT__H__

#include "lib/config.h"

#ifdef HAVE_LAPACK

#include <stdio.h>

#include "preprocessor/SimplePreprocessor.h"
#include "features/Features.h"
#include "lib/common.h"

namespace shogun
{
enum ECutoffType
{
	THRESHOLD,
	VARIANCE_EXPLAINED,
	FIXED_NUMBER
};

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
class CPCACut : public CSimplePreprocessor<float64_t>
{
	public:
		/** constructor
		 *
		 * @param do_whitening do whitening
		 * @param type of cutoff
		 * @param thresh threshold
		 */
		CPCACut(bool do_whitening=false, ECutoffType cutoff_type=THRESHOLD, float64_t thresh=1e-6);
		virtual ~CPCACut();

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

		/** get transformation matrix, i.e. eigenvectors (potentially scaled if
		 * do_whitening is true
		 *
		 * @param dst destination to store matrix in
		 * @param num_feat number of features (rows of matrix)
		 * @param num_new_dim number of dimensions after cutoff threshold
		 *
		 */
		void get_transformation_matrix(float64_t** dst, int32_t* num_feat, int32_t* num_new_dim);

		/** get eigenvalues of PCA
		 *
		 * @param dst destination to store matrix in
		 * @param num_new_dim number of dimensions after cutoff threshold
		 *
		 */
		void get_eigenvalues(float64_t** dst, int32_t* num_new_dim);

		/** get mean vector of original data
		 *
		 * @param dst destination to store matrix in
		 * @param num_feat number of features
		 *
		 */
		void get_mean(float64_t** dst, int32_t* num_feat);

		/** @return object name */
		virtual inline const char* get_name() const { return "PCACut"; }

		/// return a type of preprocessor
		virtual inline EPreprocessorType get_type() const { return P_PCACUT; }

	protected:
		void init();

	protected:
		/** T */
		double* T ;
		/** num dim */
		int32_t num_dim;
		/** num old dim */
		int32_t num_old_dim;

		/** mean */
		float64_t *mean ;
		/** length of mean vector */
		int32_t length_mean;

		/** eigenvalues */
		float64_t* eigenvalues;
		/** number of eigenvalues */
		int32_t num_eigenvalues;

		/// true when already initialized
		bool initialized;

		/** do whitening */
		bool do_whitening;
		/** Cutoff type */
		ECutoffType cutoff_type;
		/** thresh */
		float64_t thresh;
};
}
#endif
#endif
