/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Saurabh Mahindre,
 *          Sergey Lisitsyn, Evan Shelhamer, Soumyajit De, Fernando Iglesias,
 *          Bjoern Esser, parijat
 */

#include <shogun/base/progress.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/distance/Distance.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace Eigen;
using namespace shogun;


namespace shogun
{

KMeans::KMeans():KMeansBase()
{
}

KMeans::KMeans(int32_t k_i, std::shared_ptr<Distance> d_i, bool use_kmpp_i):KMeansBase(k_i, d_i, use_kmpp_i)
{
}

KMeans::KMeans(int32_t k_i, std::shared_ptr<Distance> d_i, SGMatrix<float64_t> centers_i):KMeansBase(k_i, d_i, centers_i)
{
}

KMeans::~KMeans()
{
}

void KMeans::Lloyd_KMeans(SGMatrix<float64_t> centers, int32_t num_centers)
{
	auto lhs =
		std::dynamic_pointer_cast<DenseFeatures<float64_t>>(distance->get_lhs());

	int32_t lhs_size=lhs->get_num_vectors();
	int32_t dim=lhs->get_num_features();

	auto rhs_cache = distance->get_rhs();

	SGVector<int32_t> cluster_assignments=SGVector<int32_t>(lhs_size);
	cluster_assignments.zero();

	/* Weights : Number of points in each cluster */
	SGVector<int64_t> weights_set(num_centers);
	weights_set.zero();
	/* Initially set all weights for zeroth cluster, Changes in assignement step */
	weights_set[0]=lhs_size;

	distance->precompute_lhs();

	int32_t changed=1;

	for (auto iter : SG_PROGRESS(range(max_iter)))
	{
		if (iter==max_iter-1)
			SG_SWARNING("KMeans clustering has reached maximum number of ( %d ) iterations without having converged. \
				   	Terminating. \n", iter)

		changed=0;
		auto rhs_mus = std::make_shared<DenseFeatures<float64_t>>(centers.clone());
		distance->replace_rhs(rhs_mus);

#pragma omp parallel for firstprivate(lhs_size, dim, num_centers) \
		shared(centers, cluster_assignments, weights_set) \
		reduction(+:changed) if (!fixed_centers)
		/* Assigment step : Assign each point to nearest cluster */
		for (int32_t i=0; i<lhs_size; i++)
		{
			const int32_t cluster_assignments_i=cluster_assignments[i];
			int32_t min_cluster, j;
			float64_t min_dist, dist;

			min_cluster=0;
		   	min_dist=distance->distance(i,0);
			for (j=1; j<num_centers; j++)
			{
				dist=distance->distance(i,j);
				if (dist<min_dist)
				{
					min_dist=dist;
					min_cluster=j;
				}
			}

			if (min_cluster!=cluster_assignments_i)
			{
				changed++;
#pragma omp atomic
				++weights_set[min_cluster];
#pragma omp atomic
				--weights_set[cluster_assignments_i];

				if(fixed_centers)
				{
					SGVector<float64_t>vec=lhs->get_feature_vector(i);
					float64_t temp_min = 1.0 / weights_set[min_cluster];

					/* mu_new = mu_old + (x - mu_old)/(w) */
					for (j=0; j<dim; j++)
					{
						centers(j, min_cluster)+=
							(vec[j]-centers(j, min_cluster))*temp_min;
					}

					lhs->free_feature_vector(vec, i);

					/* mu_new = mu_old - (x - mu_old)/(w-1) */
					/* if weights_set(j)~=0 */
					if (weights_set[cluster_assignments_i]!=0)
					{
						float64_t temp_i = 1.0 / weights_set[cluster_assignments_i];
						SGVector<float64_t>vec1=lhs->get_feature_vector(i);

						for (j=0; j<dim; j++)
						{
							centers(j, cluster_assignments_i)-=
								(vec1[j]-centers(j, cluster_assignments_i))*temp_i;
						}
						lhs->free_feature_vector(vec1, i);
					}
					else
					{
						/*  mus(:,j)=zeros(dim,1) ; */
						for (j=0; j<dim; j++)
							centers(j, cluster_assignments_i)=0;
					}

				}

				cluster_assignments[i] = min_cluster;
			}
		}
		if(changed==0)
			break;

		/* Update Step : Calculate new means */
		if (!fixed_centers)
		{
			/* mus=zeros(dim, num_centers) ; */
			centers.zero();

			for (int32_t i=0; i<lhs_size; i++)
			{
				int32_t cluster_i=cluster_assignments[i];

				auto vec = lhs->get_feature_vector(i);
				linalg::add_col_vec(centers, cluster_i, vec, centers);
				lhs->free_feature_vector(vec, i);
			}

			for (int32_t i=0; i<num_centers; i++)
			{
				if (weights_set[i]!=0)
				{
					auto col = centers.get_column(i);
					linalg::scale(col, col, 1.0 / weights_set[i]);
				}
			}
		}

		observe<SGMatrix<float64_t>>(iter, "mus");

		if (iter%(max_iter/10) == 0)
			SG_SINFO("Iteration[%d/%d]: Assignment of %i patterns changed.\n", iter, max_iter, changed)
	}
	distance->reset_precompute();
	distance->replace_rhs(rhs_cache);


}

bool KMeans::train_machine(std::shared_ptr<Features> data)
{
	initialize_training(data);
	Lloyd_KMeans(mus, k);
	compute_cluster_variances();
	return true;
}

}

