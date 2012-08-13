/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/features/DataGenerator.h>
#include <shogun/mathematics/Math.h>
#include <shogun/distributions/Gaussian.h>

using namespace shogun;

CDataGenerator::CDataGenerator() : CSGObject()
{
	init();
}

CDataGenerator::~CDataGenerator()
{

}

void CDataGenerator::init()
{
}

SGMatrix<float64_t> CDataGenerator::generate_mean_data(index_t m,
		index_t dim, float64_t mean_shift,
		SGMatrix<float64_t> target)
{
	/* evtl. allocate space */
	SGMatrix<float64_t> result=SGMatrix<float64_t>::get_allocated_matrix(
			dim, 2*m, target);

	/* fill matrix with normal data */
	for (index_t i=0; i<2*m; ++i)
	{
		for (index_t j=0; j<dim; ++j)
			result(j,i)=CMath::randn_double();

		/* mean shift for second half */
		if (i>=m)
			result(0,i)+=mean_shift;
	}

	return result;
}

SGMatrix<float64_t> CDataGenerator::generate_sym_mix_gauss(index_t m,
		float64_t d, float64_t angle, SGMatrix<float64_t> target)
{
	/* evtl. allocate space */
	SGMatrix<float64_t> result=SGMatrix<float64_t>::get_allocated_matrix(
			2, m, target);

	/* rotation matrix */
	SGMatrix<float64_t> rot=SGMatrix<float64_t>(2,2);
	rot(0, 0)=CMath::cos(angle);
	rot(0, 1)=-CMath::sin(angle);
	rot(1, 0)=CMath::sin(angle);
	rot(1, 1)=CMath::cos(angle);

	/* generate signal in each dimension which is an equal mixture of two
	 * Gaussians */
	for (index_t i=0; i<m; ++i)
	{
		result(0,i)=CMath::randn_double() + (CMath::random(0, 1) ? d : -d);
		result(1,i)=CMath::randn_double() + (CMath::random(0, 1) ? d : -d);
	}

	/* rotate result */
	if (angle)
		result=SGMatrix<float64_t>::matrix_multiply(rot, result);

	return result;
}

SGMatrix<float64_t> CDataGenerator::generate_gaussians(index_t m, index_t n, index_t dim)
{
	/* evtl. allocate space */
	SGMatrix<float64_t> result =
		SGMatrix<float64_t>::get_allocated_matrix(dim, n*m);

	float64_t grid_distance = 5.0;
	for (index_t i = 0; i < n; ++i)
	{
		SGVector<float64_t> mean(dim);
		SGMatrix<float64_t> cov = SGMatrix<float64_t>::create_identity_matrix(dim, 1.0);

		mean.zero();
		for (index_t k = 0; k < dim; ++k)
		{
			mean[k] = (i+1)*grid_distance;
			if (k % (i+1) == 0)
				mean[k] *= -1;
		}
		CGaussian* g = new CGaussian(mean, cov, DIAG);
		for (index_t j = 0; j < m; ++j)
		{
			SGVector<float64_t> v = g->sample();
			memcpy((result.matrix+j*result.num_rows+i*m*dim), v.vector, dim*sizeof(float64_t));
		}

		SG_UNREF(g);
	}

	return result;
}

