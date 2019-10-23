/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Viktor Gal, Weijie Lin, Sergey Lisitsyn,
 *          Bjoern Esser, Soeren Sonnenburg, Evangelos Anagnostopoulos
 */

#include <shogun/lib/config.h>

#include <shogun/features/DataGenerator.h>
#include <shogun/mathematics/Math.h>
#include <shogun/distributions/Gaussian.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/RandomNamespace.h>
#include <shogun/mathematics/UniformIntDistribution.h>
#include <shogun/mathematics/UniformRealDistribution.h>

#include <random>

using namespace shogun;

DataGenerator::DataGenerator() : SGObject()
{
	init();
}

DataGenerator::~DataGenerator()
{

}

void DataGenerator::init()
{
}

template <typename PRNG>
SGMatrix<float64_t> DataGenerator::generate_checkboard_data(int32_t num_classes,
		int32_t dim, int32_t num_points, float64_t overlap, PRNG& prng)
{
	int32_t points_per_class = num_points / num_classes;

	int32_t grid_size = (int32_t)std::ceil(std::sqrt((float64_t)num_classes));
	float64_t cell_size = (float64_t ) 1 / grid_size;
	SGVector<float64_t> grid_idx(dim);
	for (index_t i=0; i<dim; i++)
		grid_idx[i] = 0;

	SGMatrix<float64_t> points(dim+1, num_points);
	int32_t points_idx = 0;
	NormalDistribution<float64_t> normal_dist;
	UniformRealDistribution<float64_t> uniform_real_dist(0.0, 1.0);
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
					points(i, p) = normal_dist(prng, {class_dim_centers[i], cell_size*0.5});
					if ((points(i, p)>(grid_idx[i]+1)*cell_size) ||
							(points(i, p)<grid_idx[i]*cell_size))
					{
						if (!(uniform_real_dist(prng)<overlap))
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

template SGMatrix<float64_t> DataGenerator::generate_checkboard_data<std::mt19937_64>(int32_t num_classes,
		int32_t dim, int32_t num_points, float64_t overlap, std::mt19937_64& prng);


template <typename PRNG>
SGMatrix<float64_t> DataGenerator::generate_mean_data(index_t m,
		index_t dim, float64_t mean_shift, PRNG& prng,
		const SGMatrix<float64_t>& target)
{
	/* evtl. allocate space */
	SGMatrix<float64_t> result=SGMatrix<float64_t>::get_allocated_matrix(
			dim, 2*m, target);

	/* fill matrix with normal data */
	random::fill_array(result, NormalDistribution<float64_t>(), prng);
	/* mean shift for second half */
	for (index_t i=m; i<2*m; ++i)
	{
		result(0,i)+=mean_shift;
	}

	return result;
}

template SGMatrix<float64_t> DataGenerator::generate_mean_data<std::mt19937_64>(index_t m,
		index_t dim, float64_t mean_shift, std::mt19937_64& prng,
		const SGMatrix<float64_t>& target);


template <typename PRNG>
SGMatrix<float64_t> DataGenerator::generate_sym_mix_gauss(index_t m,
		float64_t d, float64_t angle, PRNG& prng, const SGMatrix<float64_t>& target)
{
	/* evtl. allocate space */
	SGMatrix<float64_t> result=SGMatrix<float64_t>::get_allocated_matrix(
			2, m, target);

	/* rotation matrix */
	SGMatrix<float64_t> rot=SGMatrix<float64_t>(2,2);
	rot(0, 0) = std::cos(angle);
	rot(0, 1) = -std::sin(angle);
	rot(1, 0) = std::sin(angle);
	rot(1, 1) = std::cos(angle);

	NormalDistribution<float64_t> normal_dist;
	UniformIntDistribution<int32_t> uniform_int_dist(0, 1);
	/* generate signal in each dimension which is an equal mixture of two
	 * Gaussians */
	for (index_t i=0; i<m; ++i)
	{
		result(0,i)=normal_dist(prng) + (uniform_int_dist(prng) ? d : -d);
		result(1,i)=normal_dist(prng) + (uniform_int_dist(prng) ? d : -d);
	}

	/* rotate result */
	if (angle)
		result=SGMatrix<float64_t>::matrix_multiply(rot, result);

	return result;
}

template SGMatrix<float64_t> DataGenerator::generate_sym_mix_gauss<std::mt19937_64>(index_t m,
		float64_t d, float64_t angle, std::mt19937_64& prng, const SGMatrix<float64_t>& target);


template <typename PRNG>
SGMatrix<float64_t> DataGenerator::generate_gaussians(index_t m, index_t n, index_t dim, PRNG& prng)
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
		auto g = std::make_shared<Gaussian>(mean, cov, DIAG);
		random::seed(g, prng);
		for (index_t j = 0; j < m; ++j)
		{
			SGVector<float64_t> v = g->sample();
			sg_memcpy((result.matrix+j*result.num_rows+i*m*dim), v.vector, dim*sizeof(float64_t));
		}


	}

	return result;
}

template SGMatrix<float64_t> DataGenerator::generate_gaussians<std::mt19937_64>(
	index_t m, index_t n, index_t dim, std::mt19937_64& prng);

