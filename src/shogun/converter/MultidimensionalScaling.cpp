/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/MultidimensionalScaling.h>
#ifdef HAVE_EIGEN3
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CMultidimensionalScaling::CMultidimensionalScaling() : CEmbeddingConverter()
{
	m_eigenvalues = SGVector<float64_t>();
	m_landmark_number = 3;
	m_landmark = false;

	init();
}

void CMultidimensionalScaling::init()
{
	SG_ADD(&m_eigenvalues, "eigenvalues", "eigenvalues of last embedding",
	    MS_NOT_AVAILABLE);
	SG_ADD(&m_landmark, "landmark",
	    "indicates if landmark approximation should be used", MS_NOT_AVAILABLE);
	SG_ADD(&m_landmark_number, "landmark_number",
	    "the number of landmarks for approximation", MS_AVAILABLE);
}

CMultidimensionalScaling::~CMultidimensionalScaling()
{
}

SGVector<float64_t> CMultidimensionalScaling::get_eigenvalues() const
{
	return m_eigenvalues;
}

void CMultidimensionalScaling::set_landmark_number(int32_t num)
{
	if (num<3)
		SG_ERROR("Number of landmarks should be greater than 3 to make triangulation possible while %d given.",
		         num);
	m_landmark_number = num;
}

int32_t CMultidimensionalScaling::get_landmark_number() const
{
	return m_landmark_number;
}

void CMultidimensionalScaling::set_landmark(bool landmark)
{
	m_landmark = landmark;
}

bool CMultidimensionalScaling::get_landmark() const
{
	return m_landmark;
}

const char* CMultidimensionalScaling::get_name() const
{
	return "MultidimensionalScaling";
};

CDenseFeatures<float64_t>* CMultidimensionalScaling::embed_distance(CDistance* distance)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	if (m_landmark) 
	{
		parameters.method = SHOGUN_LANDMARK_MULTIDIMENSIONAL_SCALING;
		parameters.landmark_ratio = float64_t(m_landmark_number)/distance->get_num_vec_lhs();
		if (parameters.landmark_ratio > 1.0) {
			SG_WARNING("Number of landmarks (%d) exceeds number of feature vectors (%d)",m_landmark_number,distance->get_num_vec_lhs());
			parameters.landmark_ratio = 1.0;
		}
	}
	else 
	{
		parameters.method = SHOGUN_MULTIDIMENSIONAL_SCALING;
	}
	parameters.target_dimension = m_target_dim;
	parameters.distance = distance;
	CDenseFeatures<float64_t>* embedding = tapkee_embed(parameters);
	return embedding;
}

CFeatures* CMultidimensionalScaling::apply(CFeatures* features)
{
	SG_REF(features);
	ASSERT(m_distance)

	m_distance->init(features,features);
	CDenseFeatures<float64_t>* embedding = embed_distance(m_distance);
	m_distance->remove_lhs_and_rhs();

	SG_UNREF(features);
	return (CFeatures*)embedding;
}

#endif /* HAVE_EIGEN3 */
