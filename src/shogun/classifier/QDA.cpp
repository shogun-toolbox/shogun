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
#include <shogun/machine/Machine.h>

using namespace shogun;

CQDA::CQDA(float64_t tolerance, bool store_covs)
: CMachine(), m_tolerance(tolerance), m_store_covs(store_covs), m_num_classes(0)
{
	init();
}

CQDA::CQDA(CSimpleFeatures<float64_t>* traindat, CLabels* trainlab, float64_t tolerance, bool store_covs)
: CMachine(), m_tolerance(tolerance), m_store_covs(store_covs), m_num_classes(0)
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
	m_parameters->add((CSGObject**) &m_covs, "m_covs", "Covariance matrices list");
	m_parameters->add((CSGObject**) &m_scalings, "m_scalings", "Scaling vectors list");
	m_parameters->add((CSGObject**) &m_rotations, "m_rotations", "Rotation matrices list");
	m_parameters->add((CSGObject**) &m_means, "m_means", "Mean vectors list");

	m_covs      = NULL;
	m_scalings  = NULL;
	m_rotations = NULL;
	m_means     = NULL;
	m_features  = NULL;
}

void CQDA::cleanup()
{
	if ( m_store_covs )
	{
		for ( int32_t i = 0 ; i < m_num_classes ; ++i )
			m_covs[i].destroy_matrix();

		delete[] m_covs;
	}

	for ( int32_t i = 0 ; i < m_num_classes ; ++i )
	{
		m_means[i].destroy_vector();
		m_scalings[i].destroy_vector();
		m_rotations[i].destroy_matrix();
	}

	delete[] m_means;
	delete[] m_scalings;
	delete[] m_rotations;

	m_covs      = NULL;
	m_scalings  = NULL;
	m_rotations = NULL;
	m_means     = NULL;

	m_num_classes = 0;
}

CLabels* CQDA::apply()
{
	// TODO
}

CLabels* CQDA::apply(CFeatures* data)
{
	// TODO
}

bool CQDA::train_machine(CFeatures* data)
{
	ASSERT(labels);
	if ( data )
	{
		if ( !data->has_property(FP_DOT) )
			SG_ERROR("Speficied features are not of type CDotFeatures\n");
		set_features((CDotFeatures*) data);
	}
	ASSERT(m_features);
	SGVector< int32_t > train_labels = labels->get_int_labels();
	ASSERT(train_labels.vector);

	cleanup();

	m_num_classes = labels->get_num_classes();
	int32_t num_feat = m_features->get_dim_feature_space();
	int32_t num_vec  = m_features->get_num_vectors();
	ASSERT(num_vec == train_labels.vlen);

	/* TODO too memory consuming like this?? */
	int32_t* class_idxs = SG_MALLOC(int32_t, num_vec*m_num_classes);
	// number of examples of each class
	int32_t* class_nums = SG_MALLOC(int32_t, m_num_classes);
	memset(class_nums, 0, m_num_classes*sizeof(int32_t));
	int32_t class_idx;
	int32_t i = 0, j = 0, k = 0;
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
		m_covs = new SGMatrix< float64_t >[m_num_classes];
		for ( i = 0 ; i < m_num_classes ; ++i )
			m_covs[i] = SGMatrix< float64_t >(num_feat, num_feat);
	}

	m_means     = new SGVector< float64_t >[m_num_classes];
	m_scalings  = new SGVector< float64_t >[m_num_classes];
	m_rotations = new SGMatrix< float64_t >[m_num_classes];
	for ( i = 0 ; i < m_num_classes ; ++i )
	{
		m_means[i]     = SGVector< float64_t >(num_feat);
		m_scalings[i]  = SGVector< float64_t >(num_feat);
		m_rotations[i] = SGMatrix< float64_t >(num_feat, num_feat);
	}

	CSimpleFeatures< float64_t >* rf = (CSimpleFeatures< float64_t >*) m_features;

	int32_t vlen;
	bool vfree;
	float64_t* vec;
	for ( k = 0 ; k < m_num_classes ; ++k )
	{
		SGMatrix< float64_t > buffer(class_nums[k], num_feat);
		m_means[k].zero();
		for ( i = 0 ; i < class_nums[k] ; ++i )
		{
			vec = rf->get_feature_vector(class_idxs[k*num_vec + i], vlen, vfree);
			ASSERT(vec);

			for ( j = 0 ; j < vlen ; ++j )
			{
				m_means[k][j] += vec[j];
				buffer[i + j*class_nums[k]] = vec[j];
			}

			rf->free_feature_vector(vec, class_idxs[k*num_vec + i], vfree);
		}
		
		for ( j = 0 ; j < num_feat ; ++j )
			m_means[k][j] /= class_nums[k];

		for ( i = 0 ; i < class_nums[k] ; ++i )
			for ( j = 0 ; j < num_feat ; ++j )
				buffer[i + j*class_nums[k]] -= m_means[k][j];

		/* calling external lib, buffer = U * S * V^T, U is not interesting here */
		char jobu = 'N', jobvt = 'A';
		int m = class_nums[k], n = num_feat;
		int lda = m, ldu = m, ldvt = n;
		int info = -1;
		wrap_dgesvd(jobu, jobvt, m, n, buffer.matrix, lda, m_scalings[k].vector,
			NULL, ldu, m_rotations[k].matrix, ldvt, &info);
		ASSERT(info == 0);
		buffer.destroy_matrix();

		CMath::vector_multiply(m_scalings[k].vector, m_scalings[k].vector,
				m_scalings[k].vector, m_scalings[k].vlen);
		CMath::scale_vector(1.0/(m-1), m_scalings[k].vector, m_scalings[k].vlen);

		CMath::transpose_matrix(m_rotations[k].matrix, n, n);

		// TODO compute and store covs if necessary
	}

#ifdef DEBUG_QDA
	SG_PRINT(">>> QDA machine trained with %d classes\n", m_num_classes);

	SG_PRINT("\n>>> Displaying means ...\n");
	for ( k = 0 ; k < m_num_classes ; ++k )
		m_means[k].display_vector();

	SG_PRINT("\n>>> Displaying scalings ...\n");
	for ( k = 0 ; k < m_num_classes ; ++k )
		m_scalings[k].display_vector();

	SG_PRINT("\n>>> Displaying rotations ... \n");
	for ( k = 0 ; k < m_num_classes ; ++k )
		CMath::display_matrix(m_rotations[k].matrix, num_feat, num_feat);

	SG_PRINT("\n>>> Exit DEBUG_QDA\n");
#endif

	train_labels.free_vector();
	SG_FREE(class_idxs);
	SG_FREE(class_nums);
	return true;
}

#endif /* HAVE_LAPACK */
