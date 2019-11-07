/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#include <shogun/features/streaming/generators/GaussianBlobsDataGenerator.h>
#include <shogun/mathematics/UniformIntDistribution.h>
#include <shogun/mathematics/NormalDistribution.h>

using namespace shogun;

GaussianBlobsDataGenerator::GaussianBlobsDataGenerator() :
		RandomMixin<StreamingDenseFeatures<float64_t>>()
{
	init();
}

GaussianBlobsDataGenerator::GaussianBlobsDataGenerator(index_t sqrt_num_blobs,
		float64_t distance, float64_t stretch, float64_t angle) :
		RandomMixin<StreamingDenseFeatures<float64_t>>()
{
	init();
	set_blobs_model(sqrt_num_blobs, distance, stretch, angle);
}

GaussianBlobsDataGenerator::~GaussianBlobsDataGenerator()
{
}

void GaussianBlobsDataGenerator::set_blobs_model(index_t sqrt_num_blobs,
		float64_t distance, float64_t stretch, float64_t angle)
{
	m_sqrt_num_blobs=sqrt_num_blobs;
	m_distance=distance;
	m_stretch=stretch;
	m_angle=angle;

	/* precompute cholesky decomposition, start with rotation matrix */
	SGMatrix<float64_t> R(2, 2);
	R(0, 0) = std::cos(angle);
	R(0, 1) = -std::sin(angle);
	R(1, 0) = std::sin(angle);
	R(1, 1) = std::cos(angle);

	/* diagonal eigenvalue matrix */
	SGMatrix<float64_t> L(2, 2);
	L(0, 0) = std::sqrt(stretch);
	L(1, 0)=0;
	L(0, 1)=0;
	L(1, 1)=1;

	/* compute and save cholesky for sampling later on */
	m_cholesky=SGMatrix<float64_t>::matrix_multiply(R, L);
}

void GaussianBlobsDataGenerator::init()
{
	SG_ADD(&m_sqrt_num_blobs, "sqrt_num_blobs", "Number of Blobs per row");
	SG_ADD(&m_distance, "distance", "Distance between blobs");
	SG_ADD(&m_stretch, "stretch", "Stretch of blobs");
	SG_ADD(&m_angle, "angle", "Angle of Blobs");
	SG_ADD(&m_cholesky, "cholesky", "Cholesky factor of covariance matrix");

	m_sqrt_num_blobs=1;
	m_distance=0;
	m_stretch=1;
	m_angle=0;
	m_cholesky=SGMatrix<float64_t>(2, 2);
	m_cholesky(0, 0)=1;
	m_cholesky(0, 1)=0;
	m_cholesky(1, 0)=0;
	m_cholesky(1, 1)=1;

	unset_generic();
}

bool GaussianBlobsDataGenerator::get_next_example()
{
	SG_TRACE("entering GaussianBlobsDataGenerator::get_next_example()");

	/* allocate space */
	SGVector<float64_t> result=SGVector<float64_t>(2);

	UniformIntDistribution<index_t> uniform_int_dist(0, m_sqrt_num_blobs-1);
	/* sample latent distribution to compute offsets */
	index_t x_offset=uniform_int_dist(m_prng)*m_distance;
	index_t y_offset=uniform_int_dist(m_prng)*m_distance;

	NormalDistribution<float64_t> normal_dist;
	/* sample from std Gaussian */
	float64_t x=normal_dist(m_prng);
	float64_t y=normal_dist(m_prng);

	/* transform through cholesky and add offset */
	result[0]=m_cholesky(0, 0)*x+m_cholesky(0, 1)*y+x_offset;
	result[1]=m_cholesky(1, 0)*x+m_cholesky(1, 1)*y+y_offset;

	/* save example back to superclass */
	GaussianBlobsDataGenerator::current_vector=result;

	SG_TRACE("leaving GaussianBlobsDataGenerator::get_next_example()");
	return true;
}

void GaussianBlobsDataGenerator::release_example()
{
	SGVector<float64_t> temp=SGVector<float64_t>();
	GaussianBlobsDataGenerator::current_vector=temp;
}

