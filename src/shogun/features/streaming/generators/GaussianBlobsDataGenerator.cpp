/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/features/streaming/generators/GaussianBlobsDataGenerator.h>

using namespace shogun;

CGaussianBlobsDataGenerator::CGaussianBlobsDataGenerator() :
		CStreamingDenseFeatures<float64_t>()
{
	init();
}

CGaussianBlobsDataGenerator::CGaussianBlobsDataGenerator(index_t sqrt_num_blobs,
		float64_t distance, float64_t stretch, float64_t angle) :
		CStreamingDenseFeatures<float64_t>()
{
	init();
	set_blobs_model(sqrt_num_blobs, distance, stretch, angle);
}

CGaussianBlobsDataGenerator::~CGaussianBlobsDataGenerator()
{
}

void CGaussianBlobsDataGenerator::set_blobs_model(index_t sqrt_num_blobs,
		float64_t distance, float64_t stretch, float64_t angle)
{
	m_sqrt_num_blobs=sqrt_num_blobs;
	m_distance=distance;
	m_stretch=stretch;
	m_angle=angle;

	/* precompute cholesky decomposition, start with rotation matrix */
	SGMatrix<float64_t> R(2, 2);
	R(0, 0)=CMath::cos(angle);
	R(0, 1)=-CMath::sin(angle);
	R(1, 0)=CMath::sin(angle);
	R(1, 1)=CMath::cos(angle);

	/* diagonal eigenvalue matrix */
	SGMatrix<float64_t> L(2, 2);
	L(0, 0)=CMath::sqrt(stretch);
	L(1, 0)=0;
	L(0, 1)=0;
	L(1, 1)=1;

	/* compute and save cholesky for sampling later on */
	m_cholesky=SGMatrix<float64_t>::matrix_multiply(R, L);
}

void CGaussianBlobsDataGenerator::init()
{
	SG_ADD(&m_sqrt_num_blobs, "sqrt_num_blobs", "Number of Blobs per row",
				MS_NOT_AVAILABLE);
	SG_ADD(&m_distance, "distance", "Distance between blobs",
				MS_NOT_AVAILABLE);
	SG_ADD(&m_stretch, "stretch", "Stretch of blobs",
				MS_NOT_AVAILABLE);
	SG_ADD(&m_angle, "angle", "Angle of Blobs",
				MS_NOT_AVAILABLE);
	SG_ADD(&m_cholesky, "cholesky", "Cholesky factor of covariance matrix",
				MS_NOT_AVAILABLE);

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

bool CGaussianBlobsDataGenerator::get_next_example()
{
	SG_SDEBUG("entering CGaussianBlobsDataGenerator::get_next_example()\n");

	/* allocate space */
	SGVector<float64_t> result=SGVector<float64_t>(2);

	/* sample latent distribution to compute offsets */
	index_t x_offset=CMath::random(0, m_sqrt_num_blobs-1)*m_distance;
	index_t y_offset=CMath::random(0, m_sqrt_num_blobs-1)*m_distance;

	/* sample from std Gaussian */
	float64_t x=CMath::randn_double();
	float64_t y=CMath::randn_double();

	/* transform through cholesky and add offset */
	result[0]=m_cholesky(0, 0)*x+m_cholesky(0, 1)*y+x_offset;
	result[1]=m_cholesky(1, 0)*x+m_cholesky(1, 1)*y+y_offset;

	/* save example back to superclass */
	CGaussianBlobsDataGenerator::current_vector=result;

	SG_SDEBUG("leaving CGaussianBlobsDataGenerator::get_next_example()\n");
	return true;
}

void CGaussianBlobsDataGenerator::release_example()
{
	SGVector<float64_t> temp=SGVector<float64_t>();
	CGaussianBlobsDataGenerator::current_vector=temp;
}

