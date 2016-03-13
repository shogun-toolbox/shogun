/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Parijat Mazumdar
 */

#include "shogun/clustering/KMeansLloydImpl.h"
#include <shogun/distance/Distance.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

namespace shogun
{
void CKMeansLloydImpl::Lloyd_KMeans(int32_t k, CDistance* distance, int32_t max_iter, SGMatrix<float64_t> mus, 
		bool fixed_centers)
{
	CDenseFeatures<float64_t>* lhs=
		CDenseFeatures<float64_t>::obtain_from_generic(distance->get_lhs());
	
	int32_t lhs_size=lhs->get_num_vectors();
	int32_t dimensions=lhs->get_num_features();

	CDenseFeatures<float64_t>* rhs_mus=new CDenseFeatures<float64_t>(0);
	CFeatures* rhs_cache=distance->replace_rhs(rhs_mus);

	SGVector<int32_t> cluster_assignments=SGVector<int32_t>(lhs_size);
	cluster_assignments.zero();

	/* Weights : Number of points in each cluster */
	SGVector<float64_t> weights_set=SGVector<float64_t>(k);
	weights_set.zero();
	/* Initially set all weights for zeroth cluster, Changes in assignement step */
	weights_set[0]=lhs_size;

	distance->precompute_lhs();

	int32_t changed=1;
	int32_t iter;

	for(iter=0; iter<max_iter; iter++)
	{
		if (iter==max_iter-1)
			SG_SWARNING("KMeans clustering has reached maximum number of ( %d ) iterations without having converged. \
				   	Terminating. \n", iter)

		changed=0;
		rhs_mus->copy_feature_matrix(mus);

		distance->precompute_rhs();
		
		/* Assigment step : Assign each point to nearest cluster */
		for (int32_t i=0; i<lhs_size; i++)
		{ 
			const int32_t cluster_assignments_i=cluster_assignments[i];
			int32_t min_cluster, j;
			float64_t min_dist, dist;
			
			min_cluster=0;
		   	min_dist=distance->distance(i,0);
			for (j=1; j<k; j++)
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
				weights_set[min_cluster]+= 1.0;
				weights_set[cluster_assignments_i]-= 1.0;

				if(fixed_centers)
				{
					SGVector<float64_t>vec=lhs->get_feature_vector(i);
					
					/* mu_new = mu_old + (x - mu_old)/(w) */					
					for (j=0; j<dimensions; j++)
					{
						mus(j, min_cluster)+=
							(vec[j]-mus(j, min_cluster)) / weights_set[min_cluster];
					}

					lhs->free_feature_vector(vec, i);

					/* mu_new = mu_old - (x - mu_old)/(w-1) */
					/* if weights_set(j)~=0 */
					if (weights_set[cluster_assignments_i]!=0.0)
					{
						SGVector<float64_t>vec1=lhs->get_feature_vector(i);

						for (j=0; j<dimensions; j++)
						{
							mus(j, cluster_assignments_i)-=
								(vec1[j]-mus(j, cluster_assignments_i)) / weights_set[cluster_assignments_i];
						}
						lhs->free_feature_vector(vec1, i);
					}
					else
					{
						/*  mus(:,j)=zeros(dimensions,1) ; */
						for (j=0; j<dimensions; j++)
							mus(j, cluster_assignments_i)=0;
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
			/* mus=zeros(dimensions, k) ; */
			mus.zero();
			for (int32_t i=0; i<lhs_size; i++)
			{
				int32_t cluster_i=cluster_assignments[i];

				SGVector<float64_t>vec=lhs->get_feature_vector(i);

				for (int32_t j=0; j<dimensions; j++)
					mus(j, cluster_i) += vec[j];
				lhs->free_feature_vector(vec, i);
			}
		
			for (int32_t i=0; i<k; i++)
			{
				if (weights_set[i]!=0.0)
				{
					for (int32_t j=0; j<dimensions; j++)
						mus(j, i) /= weights_set[i];
				}
			}
		}
		if (iter%(max_iter/10) == 0)
			SG_SINFO("Iteration[%d/%d]: Assignment of %i patterns changed.\n", iter, max_iter, changed)
	}
	distance->reset_precompute();
	distance->replace_rhs(rhs_cache);
	delete rhs_mus;
	SG_UNREF(lhs);
}
}
