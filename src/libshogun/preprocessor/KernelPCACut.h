/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Soeren Sonnenburg
 * Copyright (C) 2011 Berlin Institute of Technology
 */

#ifndef _CKERNELPCACUT__H__
#define _CKERNELPCACUT__H__

#include "lib/config.h"

#ifdef HAVE_LAPACK

#include "preprocessor/SimplePreprocessor.h"
#include "features/Features.h"
#include "kernel/Kernel.h"
#include "lib/common.h"

namespace shogun
{
	class CFeatures;
	class CKernel;

/** @brief Preprocessor PCACut performs principial component analysis on the input
 * vectors and keeps only the n eigenvectors with eigenvalues above a certain
 * threshold.
 *
 * On preprocessing the stored covariance matrix is used to project
 * vectors into eigenspace only returning vectors of reduced dimension n.
 *
 * This is only useful if the dimensionality of the data is rather low, as the
 * covariance matrix is of size num_feat*num_feat. Note that vectors don't have
 * to have zero mean as it is substracted.
 */
class CKernelPCACut : public CSimplePreprocessor<float64_t>
{
	public:
		/** default constructor
		 */
		CKernelPCACut();

		/** constructor
		 *
		 * @param thresh threshold
		 */
		CKernelPCACut(CKernel* k, float64_t thresh=1e-6);
		virtual ~CKernelPCACut();

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

		void set_kernel(CKernel* k)
		{
			kernel=k;
			SG_REF(k);
		}

		/** get transformation matrix, i.e. eigenvectors
		 *
		 * @param dst destination to store matrix in
		 * @param num_feat number of features (rows of matrix)
		 * @param num_new_dim number of dimensions after cutoff threshold
		 *
		 */
		void get_transformation_matrix(float64_t** dst, int32_t* num_feat, int32_t* num_new_dim);

		/** get bias of KPCA
		 *
		 * @param dst destination to store matrix in
		 * @param num_new_dim number of dimensions after cutoff threshold
		 *
		 */
		void get_bias(float64_t** dst, int32_t* num_new_dim);

		/** get eigenvalues of KPCA
		 *
		 * @param dst destination to store matrix in
		 * @param num_new_dim number of dimensions after cutoff threshold
		 *
		 */
		void get_eigenvalues(float64_t** dst, int32_t* num_new_dim);


		/** @return object name */
		virtual inline const char* get_name() const { return "KernelPCACut"; }

		/// return a type of preprocessor
		virtual inline EPreprocessorType get_type() const { return P_KERNELPCACUT; }

	protected:
		/** T */
		double* T ;
		int32_t rows_T;
		int32_t cols_T;


		float64_t* bias;
		int32_t bias_len;

		/** eigenvalues */
		float64_t* eigenvalues;
		/** number of eigenvalues */
		int32_t num_eigenvalues;


		/// true when already initialized
		bool initialized;

		/** thresh */
		float64_t thresh;

		CKernel* kernel;
};
}
#endif
#endif
