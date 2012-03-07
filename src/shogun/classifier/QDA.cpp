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
	// TODO
}

#endif /* HAVE_LAPACK */
