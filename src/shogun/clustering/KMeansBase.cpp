/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre, Heiko Strathmann, Pan Deng, Viktor Gal
 */

#include <shogun/base/Parallel.h>
#include <shogun/clustering/KMeansBase.h>
#include <shogun/distance/Distance.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/Labels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/mathematics/RandomNamespace.h>

#include <utility>

using namespace shogun;
using namespace Eigen;

KMeansBase::KMeansBase()
: RandomMixin<DistanceMachine>()
{
	init();
}

KMeansBase::KMeansBase(int32_t k_, std::shared_ptr<Distance> d, bool use_kmpp)
: RandomMixin<DistanceMachine>()
{
	init();
	k=k_;
	set_distance(std::move(d));
	use_kmeanspp=use_kmpp;
}

KMeansBase::KMeansBase(int32_t k_i, std::shared_ptr<Distance> d_i, SGMatrix<float64_t> centers_i)
: RandomMixin<DistanceMachine>()
{
	init();
	k = k_i;
	set_distance(std::move(d_i));
	set_initial_centers(centers_i);
}

KMeansBase::~KMeansBase()
{
}

void KMeansBase::set_initial_centers(SGMatrix<float64_t> centers)
{
	auto lhs=distance->get_lhs()->as<DenseFeatures<float64_t>>();
	dimensions=lhs->get_num_features();
	require(centers.num_cols == k,
			"Expected {} initial cluster centers, got {}", k, centers.num_cols);
	require(centers.num_rows == dimensions,
			"Expected {} dimensionional cluster centers, got {}", dimensions, centers.num_rows);
	mus_initial = centers;

}

void KMeansBase::set_random_centers()
{
	mus.zero();
	auto lhs=
		distance->get_lhs()->as<DenseFeatures<float64_t>>();
	int32_t lhs_size=lhs->get_num_vectors();

	SGVector<int32_t> temp=SGVector<int32_t>(lhs_size);
	SGVector<int32_t>::range_fill_vector(temp, lhs_size, 0);
	random::shuffle(temp, m_prng);

	for (int32_t i=0; i<k; i++)
	{
		const int32_t cluster_center_i=temp[i];
		SGVector<float64_t> vec=lhs->get_feature_vector(cluster_center_i);

		for (int32_t j=0; j<dimensions; j++)
			mus(j,i)=vec[j];

		lhs->free_feature_vector(vec, cluster_center_i);
	}

	observe<SGMatrix<float64_t>>(0, "mus");
}

void KMeansBase::compute_cluster_variances()
{
	/* compute the ,,variances'' of the clusters */
	for (int32_t i=0; i<k; i++)
	{
		float64_t rmin1=0;
		float64_t rmin2=0;

		bool first_round=true;

		for (int32_t j=0; j<k; j++)
		{
			if (j!=i)
			{
				int32_t l;
				float64_t dist = 0;

				for (l=0; l<dimensions; l++)
				{
					dist+=Math::sq(
							mus.matrix[i*dimensions+l]
									-mus.matrix[j*dimensions+l]);
				}

				if (first_round)
				{
					rmin1=dist;
					rmin2=dist;
					first_round=false;
				}
				else
				{
					if ((dist<rmin2) && (dist>=rmin1))
						rmin2=dist;

					if (dist<rmin1)
					{
						rmin2=rmin1;
						rmin1=dist;
					}
				}
			}
		}

		R.vector[i] = (0.7 * std::sqrt(rmin1) + 0.3 * std::sqrt(rmin2));
	}
}

void KMeansBase::initialize_training(const std::shared_ptr<Features>& data)
{
	require(distance, "Distance is not provided");
	require(
	    distance->get_feature_type() == F_DREAL,
	    "Distance's features type ({}) should be of type REAL ({})");
	require(
	    max_iter > 0,
	    "The number of iterations provided ({}) must be greater than 0",
	    max_iter);
	require(
	    k > 0, "The number of clusters provided ({}) must be greater than 0",
	    k);

	if (data)
		distance->init(data, data);

	auto lhs=
		distance->get_lhs()->as<DenseFeatures<float64_t>>();

	require(lhs, "Lhs features of distance not provided");
	int32_t lhs_size=lhs->get_num_vectors();
	dimensions=lhs->get_num_features();
	const int32_t centers_size=dimensions*k;

	require(lhs_size>0, "Lhs features should not be empty");
	require(dimensions>0, "Lhs features should have more than zero dimensions");

	/* if kmeans++ to be used */
	if (use_kmeanspp)
		mus_initial=kmeanspp();

	R=SGVector<float64_t>(k);

	mus=SGMatrix<float64_t>(dimensions, k);
	/* cluster_centers=zeros(dimensions, k) ; */
	memset(mus.matrix, 0, sizeof(float64_t)*centers_size);

	if (mus_initial.matrix)
	{
		mus = mus_initial;
		observe<SGMatrix<float64_t>>(0, "mus");
	}
	else
	{
		set_random_centers();
	}
}

bool KMeansBase::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool KMeansBase::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

SGMatrix<float64_t> KMeansBase::get_cluster_centers() const
{
	return mus;
}

SGMatrix<float64_t> KMeansBase::kmeanspp()
{
	int32_t lhs_size;
	auto lhs=distance->get_lhs()->as<DenseFeatures<float64_t>>();
	lhs_size=lhs->get_num_vectors();

	SGMatrix<float64_t> centers=SGMatrix<float64_t>(dimensions, k);
	centers.zero();
	SGVector<float64_t> min_dist=SGVector<float64_t>(lhs_size);
	min_dist.zero();

	UniformIntDistribution<int32_t> uniform_int_dist(0, lhs_size-1);
	/* First center is chosen at random */
	int32_t mu=uniform_int_dist(m_prng);
	SGVector<float64_t> mu_first=lhs->get_feature_vector(mu);
	for(int32_t j=0; j<dimensions; j++)
		centers(j, 0)=mu_first[j];

	distance->precompute_lhs();
	distance->precompute_rhs();
#pragma omp parallel for \
	default(none) shared(min_dist, mu, lhs_size) \
	schedule(static, CPU_CACHE_LINE_SIZE_BYTES)
	for(int32_t i=0; i<lhs_size; i++)
		min_dist[i]=Math::sq(distance->distance(i, mu));
#ifdef HAVE_LINALG
	float64_t sum=linalg::vector_sum(min_dist);
#else //HAVE_LINALG
	Eigen::Map<VectorXd> eigen_min_dist(min_dist.vector, min_dist.vlen);
	float64_t sum=eigen_min_dist.sum();
#endif //HAVE_LINALG
	int32_t n_rands = 2 + int32_t(std::log(k));

	UniformRealDistribution<float64_t> uniform_real_dist(0.0, 1.0);
	/* Choose centers with weighted probability */
	for(int32_t i=1; i<k; i++)
	{
		int32_t best_center=0;
		float64_t best_sum=-1.0;
		SGVector<float64_t> best_min_dist=SGVector<float64_t>(lhs_size);

		/* local tries for best center */
		for(int32_t trial=0; trial<n_rands; trial++)
		{
			float64_t temp_sum=0.0;
			float64_t temp_dist=0.0;
			SGVector<float64_t> temp_min_dist=SGVector<float64_t>(lhs_size);
			int32_t new_center=0;
			float64_t prob=uniform_real_dist(m_prng);
			prob=prob*sum;

			for(int32_t j=0; j<lhs_size; j++)
			{
				temp_sum+=min_dist[j];
				if (prob <= temp_sum)
				{
					new_center=j;
					break;
				}
			}

#pragma omp parallel for default(none) \
			private(temp_dist) shared(temp_min_dist, min_dist, lhs_size, new_center) \
			schedule(static, CPU_CACHE_LINE_SIZE_BYTES)
			for(int32_t j=0; j<lhs_size; j++)
			{
				temp_dist=Math::sq(distance->distance(j, new_center));
				temp_min_dist[j]=Math::min(temp_dist, min_dist[j]);
			}

#ifdef HAVE_LINALG
			temp_sum=linalg::vector_sum(temp_min_dist);
#else //HAVE_LINALG
			Eigen::Map<VectorXd> eigen_temp_sum(temp_min_dist.vector, temp_min_dist.vlen);
			temp_sum=eigen_temp_sum.sum();
#endif //HAVE_LINALG
			if ((temp_sum<best_sum) || (best_sum<0))
			{
				best_sum=temp_sum;
				best_min_dist=temp_min_dist;
				best_center=new_center;
			}
		}

		SGVector<float64_t> vec=lhs->get_feature_vector(best_center);
		for(int32_t j=0; j<dimensions; j++)
			centers(j, i)=vec[j];
		sum=best_sum;
		min_dist=best_min_dist;
	}

	distance->reset_precompute();

	return centers;
}

void KMeansBase::init()
{
	max_iter = 300;
	k = 8;
	dimensions = 0;
	fixed_centers = false;
	use_kmeanspp = false;
	SG_ADD(
	    &max_iter, "max_iter", "Maximum number of iterations",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &k, "k", "k, the number of clusters",
	    ParameterProperties::HYPER | ParameterProperties::CONSTRAIN,
	    SG_CONSTRAINT(positive<>()));
	SG_ADD(
	    &dimensions, "dimensions", "Dimensions of data",
	    ParameterProperties::READONLY);
	SG_ADD(
	    &fixed_centers, "fixed_centers", "Whether to use fixed centers",
	    ParameterProperties::HYPER | ParameterProperties::SETTING);
	SG_ADD(&R, "radiuses", "Cluster radiuses", ParameterProperties::MODEL);
	SG_ADD(
	    &use_kmeanspp, "kmeanspp", "Whether to use kmeans++",
	    ParameterProperties::HYPER | ParameterProperties::SETTING);
	SG_ADD(&mus, "mus", "Cluster centers", ParameterProperties::MODEL);

	watch_method("cluster_centers", &KMeansBase::get_cluster_centers);
}
