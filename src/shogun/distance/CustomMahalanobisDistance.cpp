/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Fernando Iglesias, Viktor Gal, Heiko Strathmann
 */

#include <shogun/distance/CustomMahalanobisDistance.h>

#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CustomMahalanobisDistance::CustomMahalanobisDistance() : RealDistance()
{
	register_params();
}

CustomMahalanobisDistance::CustomMahalanobisDistance(std::shared_ptr<Features> l, std::shared_ptr<Features> r, SGMatrix<float64_t> m)
: RealDistance()
{
	register_params();
	RealDistance::init(l, r);
	m_mahalanobis_matrix = m;
}

void CustomMahalanobisDistance::register_params()
{
	SG_ADD(&m_mahalanobis_matrix, "m_mahalanobis_matrix", "Mahalanobis matrix");
}

CustomMahalanobisDistance::~CustomMahalanobisDistance()
{
	cleanup();
}

void CustomMahalanobisDistance::cleanup()
{
}

const char* CustomMahalanobisDistance::get_name() const
{
	return "CustomMahalanobisDistance";
}

EDistanceType CustomMahalanobisDistance::get_distance_type()
{
	return D_CUSTOMMAHALANOBIS;
}

float64_t CustomMahalanobisDistance::compute(int32_t idx_a, int32_t idx_b)
{
	// Get feature vectors that will be used to compute the distance; casts
	// are safe, features are checked to be dense in DenseDistance::init
	SGVector<float64_t> avec = std::dynamic_pointer_cast<DenseFeatures<float64_t>>(lhs)->get_feature_vector(idx_a);
	SGVector<float64_t> bvec = std::dynamic_pointer_cast<DenseFeatures<float64_t>>(rhs)->get_feature_vector(idx_b);

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

