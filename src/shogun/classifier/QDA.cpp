/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/lib/common.h>

#ifdef HAVE_LAPACK

#include <shogun/classifier/QDA.h>
#include <shogun/features/Features.h>
#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/lapack.h>

using namespace shogun;

CQDA::CQDA(float64_t tolerance, bool store_covs)
: CMachine(), m_tolerance(tolerance), m_store_covs(store_covs), m_num_classes(0), m_dim(0)
{
	init();
}

CQDA::CQDA(CSimpleFeatures<float64_t>* traindat, CLabels* trainlab, float64_t tolerance, bool store_covs)
: CMachine(), m_tolerance(tolerance), m_store_covs(store_covs), m_num_classes(0), m_dim(0)
{
	init();
	set_features(traindat);
	set_labels(trainlab);
}

CQDA::~CQDA()
{
	SG_UNREF(m_features);

	cleanup();
}

void CQDA::init()
{
	m_parameters->add(&m_tolerance, "m_tolerance", "Tolerance member.");
	m_parameters->add(&m_store_covs, "m_store_covs", "Store covariances member");
	m_parameters->add((CSGObject**) &m_features, "m_features", "Feature object.");
	m_parameters->add(&m_means, "m_means", "Mean vectors list");
	m_parameters->add(&m_slog, "m_slog", "Vector used in classification");

	//TODO include SGNDArray objects for serialization

	m_features  = NULL;
}

void CQDA::cleanup()
{
	if ( m_store_covs )
		m_covs.destroy_ndarray();

	m_covs.free_ndarray();
	m_M.free_ndarray();
	m_means.free_matrix();
	m_slog.free_vector();

	m_num_classes = 0;
}

CLabels* CQDA::apply()
{
	if ( !m_features )
		return NULL;

	int32_t num_vecs = m_features->get_num_vectors();
	ASSERT(num_vecs > 0);
	ASSERT( m_dim == m_features->get_dim_feature_space() );

	CSimpleFeatures< float64_t >* rf = (CSimpleFeatures< float64_t >*) m_features;

	SGMatrix< float64_t > X(num_vecs, m_dim);
	SGMatrix< float64_t > A(num_vecs, m_dim);
	SGVector< float64_t > norm2(num_vecs*m_num_classes);

	norm2.zero();

	int i, j, k, vlen;
	bool vfree;
	float64_t* vec;
	for ( k = 0 ; k < m_num_classes ; ++k )
	{
		// X = features - means
		for ( i = 0 ; i < num_vecs ; ++i )
		{
			vec = rf->get_feature_vector(i, vlen, vfree);
			ASSERT(vec);

			for ( j = 0 ; j < m_dim ; ++j )
				X[i + j*num_vecs] = vec[j] - m_means[k*m_dim + j];

			rf->free_feature_vector(vec, i, vfree);

		}

		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, num_vecs, m_dim, 
			m_dim, 1.0, X.matrix, num_vecs, m_M.get_matrix(k), m_dim, 0.0, 
			A.matrix, num_vecs);

		for ( i = 0 ; i < num_vecs ; ++i )
			for ( j = 0 ; j < m_dim ; ++j )
				norm2[i + k*num_vecs] += CMath::sq(A[i + j*num_vecs]);

#ifdef DEBUG_QDA
	CMath::display_matrix(A.matrix, num_vecs, m_dim, "A");
#endif
	}

	for ( i = 0 ; i < num_vecs ; ++i )
		for ( k = 0 ; k < m_num_classes ; ++k )
		{
			norm2[i + k*num_vecs] += m_slog[k];
			norm2[i + k*num_vecs] *= -0.5;
		}

	CLabels* out = new CLabels(num_vecs);

	for ( i = 0 ; i < num_vecs ; ++i )
		out->set_label(i, CMath::arg_max(norm2.vector+i, num_vecs, m_num_classes));

#ifdef DEBUG_QDA
	CMath::display_matrix(norm2.vector, num_vecs, m_num_classes, "norm2");
	CMath::display_vector(out->get_labels().vector, num_vecs, "Labels");
#endif

	norm2.destroy_vector();
	A.destroy_matrix();
	X.destroy_matrix();

	return out;
}

CLabels* CQDA::apply(CFeatures* data)
{
	if ( !data )
		SG_ERROR("No features specified\n");
	if ( !data->has_property(FP_DOT) )
		SG_ERROR("Specified features are not of type CDotFeatures\n");

	set_features((CDotFeatures*) data);
	return apply();
}

bool CQDA::train_machine(CFeatures* data)
{
	if ( !m_labels )
		SG_ERROR("No labels allocated in QDA training\n");

	if ( data )
	{
		if ( !data->has_property(FP_DOT) )
			SG_ERROR("Speficied features are not of type CDotFeatures\n");
		set_features((CDotFeatures*) data);
	}
	if ( !m_features )
		SG_ERROR("No features allocated in QDA training\n");
	SGVector< int32_t > train_labels = m_labels->get_int_labels();
	if ( !train_labels.vector )
		SG_ERROR("No train_labels allocated in QDA training\n");

	cleanup();

	m_num_classes = m_labels->get_num_classes();
	m_dim = m_features->get_dim_feature_space();
	int32_t num_vec  = m_features->get_num_vectors();
	if ( num_vec != train_labels.vlen )
		SG_ERROR("Dimension mismatch between features and labels in QDA training");

	int32_t* class_idxs = SG_MALLOC(int32_t, num_vec*m_num_classes);
	// number of examples of each class
	int32_t* class_nums = SG_MALLOC(int32_t, m_num_classes);
	memset(class_nums, 0, m_num_classes*sizeof(int32_t));
	int32_t class_idx;
	int32_t i, j, k;
	for ( i = 0 ; i < train_labels.vlen ; ++i )
	{
		class_idx = train_labels.vector[i];

		if ( class_idx < 0 || class_idx >= m_num_classes )
		{
			SG_ERROR("found label out of {0, 1, 2, ..., num_classes-1}...");
			return false;
		}
		else
		{
			class_idxs[ class_idx*num_vec + class_nums[class_idx]++ ] = i;
		}
	}

	for ( i = 0 ; i < m_num_classes ; ++i )
	{
		if ( class_nums[i] <= 0 )
		{
			SG_ERROR("What? One class with no elements\n");
			return false;
		}
	}

	if ( m_store_covs )
	{
		// cov_dims will be free in m_covs.destroy_ndarray()
		index_t * cov_dims = SG_MALLOC(index_t, 3);
		cov_dims[0] = m_dim;
		cov_dims[1] = m_dim;
		cov_dims[2] = m_num_classes;
		m_covs = SGNDArray< float64_t >(cov_dims, 3, true);
	}

	m_means = SGMatrix< float64_t >(m_dim, m_num_classes, true);
	SGMatrix< float64_t > scalings  = SGMatrix< float64_t >(m_dim, m_num_classes);

	// rot_dims will be freed in rotations.destroy_ndarray()
	index_t* rot_dims = SG_MALLOC(index_t, 3);
	rot_dims[0] = m_dim;
	rot_dims[1] = m_dim;
	rot_dims[2] = m_num_classes;
	SGNDArray< float64_t > rotations = SGNDArray< float64_t >(rot_dims, 3);

	CSimpleFeatures< float64_t >* rf = (CSimpleFeatures< float64_t >*) m_features;

	m_means.zero();

	int32_t vlen;
	bool vfree;
	float64_t* vec;
	for ( k = 0 ; k < m_num_classes ; ++k )
	{
		SGMatrix< float64_t > buffer(class_nums[k], m_dim);
		for ( i = 0 ; i < class_nums[k] ; ++i )
		{
			vec = rf->get_feature_vector(class_idxs[k*num_vec + i], vlen, vfree);
			ASSERT(vec);

			for ( j = 0 ; j < vlen ; ++j )
			{
				m_means[k*m_dim + j] += vec[j];
				buffer[i + j*class_nums[k]] = vec[j];
			}

			rf->free_feature_vector(vec, class_idxs[k*num_vec + i], vfree);
		}
		
		for ( j = 0 ; j < m_dim ; ++j )
			m_means[k*m_dim + j] /= class_nums[k];

		for ( i = 0 ; i < class_nums[k] ; ++i )
			for ( j = 0 ; j < m_dim ; ++j )
				buffer[i + j*class_nums[k]] -= m_means[k*m_dim + j];

		/* calling external lib, buffer = U * S * V^T, U is not interesting here */
		char jobu = 'N', jobvt = 'A';
		int m = class_nums[k], n = m_dim;
		int lda = m, ldu = m, ldvt = n;
		int info = -1;
		float64_t * col = scalings.get_column_vector(k);
		float64_t * rot_mat = rotations.get_matrix(k);

		wrap_dgesvd(jobu, jobvt, m, n, buffer.matrix, lda, col, NULL, ldu, 
			rot_mat, ldvt, &info);
		ASSERT(info == 0);
		buffer.destroy_matrix();

		CMath::vector_multiply(col, col, col, m_dim);
		CMath::scale_vector(1.0/(m-1), col, m_dim);
		rotations.transpose_matrix(k);

		if ( m_store_covs )
		{
			SGMatrix< float64_t > M(n ,n);

			M.matrix = CMath::clone_vector(rot_mat, n*n);
			for ( i = 0 ; i < m_dim ; ++i )
				for ( j = 0 ; j < m_dim ; ++j )
					M[i + j*m_dim] *= scalings[k*m_dim + j];
			
			cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, n, n, n, 1.0,
				M.matrix, n, rot_mat, n, 0.0, m_covs.get_matrix(k), n);

			M.destroy_matrix();
		}
	}

	/* Computation of terms required for classification */

	SGVector< float32_t > sinvsqrt(m_dim);

	// M_dims will be freed in m_M.destroy_ndarray()
	index_t* M_dims = SG_MALLOC(index_t, 3);
	M_dims[0] = m_dim;
	M_dims[1] = m_dim;
	M_dims[2] = m_num_classes;
	m_M = SGNDArray< float64_t >(M_dims, 3, true);

	m_slog = SGVector< float32_t >(m_num_classes, true);
	m_slog.zero();

	index_t idx = 0;
	for ( k = 0 ; k < m_num_classes ; ++k )
	{
		for ( j = 0 ; j < m_dim ; ++j )
		{
			sinvsqrt[j] = 1.0 / CMath::sqrt(scalings[k*m_dim + j]);
			m_slog[k]  += CMath::log(scalings[k*m_dim + j]);
		}

		for ( i = 0 ; i < m_dim ; ++i )
			for ( j = 0 ; j < m_dim ; ++j )
			{
				idx = k*m_dim*m_dim + i + j*m_dim;
				m_M[idx] = rotations[idx] * sinvsqrt[j];
			}
	}

#ifdef DEBUG_QDA
	SG_PRINT(">>> QDA machine trained with %d classes\n", m_num_classes);

	SG_PRINT("\n>>> Displaying means ...\n");
	CMath::display_matrix(m_means.matrix, m_dim, m_num_classes);

	SG_PRINT("\n>>> Displaying scalings ...\n");
	CMath::display_matrix(scalings.matrix, m_dim, m_num_classes);

	SG_PRINT("\n>>> Displaying rotations ... \n");
	for ( k = 0 ; k < m_num_classes ; ++k )
		CMath::display_matrix(rotations.get_matrix(k), m_dim, m_dim);

	SG_PRINT("\n>>> Displaying sinvsqrt ... \n");
	sinvsqrt.display_vector();

	SG_PRINT("\n>>> Diplaying m_M matrices ... \n");
	for ( k = 0 ; k < m_num_classes ; ++k )
		CMath::display_matrix(m_M.get_matrix(k), m_dim, m_dim);

	SG_PRINT("\n>>> Exit DEBUG_QDA\n");
#endif

	rotations.destroy_ndarray();
	scalings.destroy_matrix();
	sinvsqrt.destroy_vector();
	train_labels.destroy_vector();
	SG_FREE(class_idxs);
	SG_FREE(class_nums);
	return true;
}

#endif /* HAVE_LAPACK */
