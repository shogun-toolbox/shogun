/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <lib/config.h>

#include <features/DataGenerator.h>
#include <mathematics/Math.h>
#include <distributions/Gaussian.h>

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

SGMatrix<float64_t> CDataGenerator::generate_checkboard_data(int32_t num_classes,
		int32_t dim, int32_t num_points, float64_t overlap)
{
	int32_t points_per_class = num_points / num_classes;

	int32_t grid_size = (int32_t ) CMath::ceil(CMath::sqrt((float64_t ) num_classes));
	float64_t cell_size = (float64_t ) 1 / grid_size;
	SGVector<float64_t> grid_idx(dim);
	for (index_t i=0; i<dim; i++)
		grid_idx[i] = 0;

	SGMatrix<float64_t> points(dim+1, num_points);
	int32_t points_idx = 0;
	for (int32_t class_idx=0; class_idx<num_classes; class_idx++)
	{
		SGVector<float64_t> class_dim_centers(dim);
		for (index_t i=0; i<dim; i++)
			class_dim_centers[i] = (grid_idx[i] * cell_size + (grid_idx[i] + 1) * cell_size) / 2;

		for (index_t p=points_idx; p<points_per_class+points_idx; p++)
		{
			for (index_t i=0; i<dim; i++)
			{
				do
				{
					points(i, p) = CMath::normal_random(class_dim_centers[i], cell_size*0.5);
					if ((points(i, p)>(grid_idx[i]+1)*cell_size) ||
							(points(i, p)<grid_idx[i]*cell_size))
					{
						if (!(CMath::random(0.0, 1.0)<overlap))
							continue;
					}
					break;
				} while (true);
			}
			points(dim, p) = class_idx;
		}
		points_idx += points_per_class;
		for (index_t i=dim-1; i>=0; i--)
		{
			grid_idx[i]++;
			if (grid_idx[i]>=grid_size)
				grid_idx[i] = 0;
			else
				break;
		}
	}
	return points;
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
#ifdef HAVE_LAPACK
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
			SG_FREE(v.vector);
		}

		SG_UNREF(g);
	}

	return result;
}
#endif /* HAVE_LAPACK */
