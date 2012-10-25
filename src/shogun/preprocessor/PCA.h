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

#ifndef PCA_H_
#define PCA_H_
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/lapack.h>
#include <stdio.h>
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/lib/common.h>

namespace shogun
{
/** mode of pca */
enum EPCAMode
{
	/** cut by threshold */
	THRESHOLD,
	/** variance explained */
	VARIANCE_EXPLAINED,
	/** keep fixed number of features */
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
class CPCA: public CDimensionReductionPreprocessor
{
	public:

		/** constructor
		 * @param do_whitening do whitening
		 * @param mode mode of pca
		 * @param thresh threshold
		 */
		CPCA(bool do_whitening=false, EPCAMode mode=FIXED_NUMBER, float64_t thresh=1e-6);

		/** destructor */
		virtual ~CPCA();

		/** initialize preprocessor from features
		 * @param features
		 */
		virtual bool init(CFeatures* features);

		/** cleanup */
		virtual void cleanup();

		/** apply preprocessor to feature matrix
		 * @param features features
		 * @return processed feature matrix
		 */
		virtual SGMatrix<float64_t> apply_to_feature_matrix(CFeatures* features);

		/** apply preprocessor to feature vector
		 * @param vector feature vector
		 * @return processed feature vector
		 */
		virtual SGVector<float64_t> apply_to_feature_vector(SGVector<float64_t> vector);

		/** get transformation matrix, i.e. eigenvectors (potentially scaled if
		 * do_whitening is true)
		 */
		SGMatrix<float64_t> get_transformation_matrix();

		/** get eigenvalues of PCA
		 */
		SGVector<float64_t> get_eigenvalues();

		/** get mean vector of original data
		 */
		SGVector<float64_t> get_mean();

		/** @return object name */
		virtual const char* get_name() const { return "PCA"; }

		/** @return a type of preprocessor */
		virtual EPreprocessorType get_type() const { return P_PCA; }

	protected:

		void init();

	protected:

		/** transformation matrix */
		SGMatrix<float64_t> m_transformation_matrix;
		/** num dim */
		int32_t num_dim;
		/** num old dim */
		int32_t num_old_dim;
		/** mean vector */
		SGVector<float64_t> m_mean_vector;
		/** eigenvalues vector */
		SGVector<float64_t> m_eigenvalues_vector;
		/** initialized */
		bool m_initialized;
		/** whitening */
		bool m_whitening;
		/** PCA mode */
		EPCAMode m_mode;
		/** thresh */
		float64_t thresh;
};
}
#endif
#endif
