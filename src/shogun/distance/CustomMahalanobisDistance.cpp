/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Fernando J. Iglesias Garcia
 * Copyright (C) 2013 Fernando J. Iglesias Garcia
 */

#ifdef HAVE_EIGEN3

#include <shogun/distance/CustomMahalanobisDistance.h>
#include <Eigen/Dense>

using namespace shogun;
using namespace Eigen;

CCustomMahalanobisDistance::CCustomMahalanobisDistance() : CRealDistance()
{
	register_params();
}

CCustomMahalanobisDistance::CCustomMahalanobisDistance(CFeatures* l, CFeatures* r, SGMatrix<float64_t> m)
: CRealDistance()
{
	register_params();
	CRealDistance::init(l, r);
	m_mahalanobis_matrix = m;
}

void CCustomMahalanobisDistance::register_params()
{
	SG_ADD(&m_mahalanobis_matrix, "m_mahalanobis_matrix", "Mahalanobis matrix", MS_NOT_AVAILABLE)
}

CCustomMahalanobisDistance::~CCustomMahalanobisDistance()
{
	cleanup();
}

void CCustomMahalanobisDistance::cleanup()
{
}

const char* CCustomMahalanobisDistance::get_name() const
{
	return "CustomMahalanobisDistance";
}

EDistanceType CCustomMahalanobisDistance::get_distance_type()
{
	return D_CUSTOMMAHALANOBIS;
}

float64_t CCustomMahalanobisDistance::compute(int32_t idx_a, int32_t idx_b)
{
	// Get feature vectors that will be used to compute the distance; casts
	// are safe, features are checked to be dense in DenseDistance::init
	SGVector<float64_t> avec = static_cast<CDenseFeatures<float64_t>*>(lhs)->get_feature_vector(idx_a);
	SGVector<float64_t> bvec = static_cast<CDenseFeatures<float64_t>*>(rhs)->get_feature_vector(idx_b);

	REQUIRE(avec.vlen == bvec.vlen, "In CCustomMahalanobisDistance::compute the "
			"feature vectors must have the same number of elements")

	// Compute the distance between the feature vectors

	// Compute the difference vector and wrap in Eigen vector
	const VectorXd dvec = Map<const VectorXd>(avec, avec.vlen) - Map<const VectorXd>(bvec, bvec.vlen);
	// Wrap Mahalanobis distance in Eigen matrix
	Map<const MatrixXd> M(m_mahalanobis_matrix.matrix, m_mahalanobis_matrix.num_rows,
			m_mahalanobis_matrix.num_cols);

	return dvec.transpose()*M*dvec;
}

#endif /* HAVE_EIGEN3 */
