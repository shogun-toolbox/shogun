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
	set_features(traindat);
	set_labels(trainlab);
	init();
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
}

CQDA::~CQDA()
{
	SG_UNREF(m_features);

	/* TODO loop here and free every SGVector/SGMatrix?? */
	if ( m_store_covs ) 
		SG_FREE(m_covs);
	SG_FREE(m_scalings);
	SG_FREE(m_rotations);
	SG_FREE(m_means);

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
	ASSERT(trains_labels.vector);

	m_num_classes = labels->get_num_classes();

	int32_t num_feat = m_features->get_dim_feature_space();
	int32_t num_vec  = m_features->get_num_vectors();
	ASSERT(num_vec == train_labels.vlen);

	/* TODO too memory consuming like this?? */
	int32_t* class_idxs = SG_MALLOC(int32_t, num_vec*m_num_classes);
	// number of examples of each class
	int32_t* class_nums = SG_MALLOC(int32_t, m_num_classes);
	memset(class_nums, 0, m_num_classes*sizeof(int32_t));
	int32_t i = 0, j = 0, k = 0;
	int32_t class_idx;
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

	/* TODO loop here and free every SGVector/SGMatrix?? */
	SG_FREE(m_covs);
	SG_FREE(m_means);
	SG_FREE(m_scalings)g_vector
	SG_FREE(m_rotations);

	if ( m_store_covs )
	{
		m_covs = SG_MALLOC(SGMatrix< float64_t >*, m_num_classes);
		for ( i = 0 ; i < m_num_classes ; ++i )
			m_covs[i] = new SGMatrix< float64_t >(num_feat, num_feat);
	}
	m_means     = SG_MALLOC(SGVector< float64_t >*, m_num_classes);
	m_scalings  = SG_MALLOC(SGVector< float64_t >*, m_num_classes);
	m_rotations = SG_MALLOC(SGMatrix< float64_t >*, m_num_classes);
	for ( i = 0 ; i < m_num_classes ; ++i )
	{
		/* TODO check the dimensions of these guys */
		m_scalings[i]  = new SGVector< float64_t >(num_feat);
		m_rotations[i] = new SGMatrix< float64_t >(num_feat, num_feat);
	}

	SGVector< float64_t > mean = SGVector< float64_t>(num_feat);
	CSimpleFeatures< float64_t >* rf = (CSimpleFeatures< float64_t >*) features;

	int32_t vlen;
	bool vfree;

	for ( i = 0 ; i < m_num_classes ; ++i )
	{
		mean.set_const(0);
		for ( j = 0 ; j < class_nums[i] ; ++j )
		{
			float64_t* vec = rf->get_feature_vector( class_idxs[i*num_vec + j] );
			ASSERT(vec);

			for ( k = 0 ; k < vlen ; ++k )
			{
				mean[k] += vec[k];
				buffer[num_feat*j + k] = vec[k];
			}

			rf->free_feature_vector(vec, class_idxs[i*num_vec + j], vfree);
		}
		
		for ( k = 0 ; k < num_feat ; ++k )
			mean[k] /= class_nums[i];

		for ( j = 0 ; j < class_nums[i] ; ++j )
		{
			for ( k = 0 ; k < num_feat ; ++k )
				buffer[num_feat*j + k] -= mean[k];
		}

		m_means[i]->vector = SGVector< float64_t >.clone_vector(mean);
	}

	train_labels.free_vector();
	SG_FREE(class_idxs);
	SG_FREE(class_nums);
}

#endif /* HAVE_LAPACK */
