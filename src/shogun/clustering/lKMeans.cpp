/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Parijat Mazumdar
 */

#include "shogun/clustering/lKMeans.h"
#include <shogun/distance/Distance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

namespace shogun
{
void Lloyd::Lloyd_KMeans(int32_t k, CDistance* distance, int32_t max_iter, SGMatrix<float64_t> mus, 
		SGVector<int32_t> ClList, SGVector<float64_t> weights_set, bool fixed_centers)
{
	CDenseFeatures<float64_t>* lhs=
		CDenseFeatures<float64_t>::obtain_from_generic(distance->get_lhs());
	int32_t XSize=lhs->get_num_vectors();
	int32_t dimensions=lhs->get_num_features();

	CDenseFeatures<float64_t>* rhs_mus=new CDenseFeatures<float64_t>(0);
	CFeatures* rhs_cache=distance->replace_rhs(rhs_mus);

	SGVector<float64_t> dists=SGVector<float64_t>(k*XSize);
	dists.zero();

	int32_t changed=1;
	int32_t iter=0;
	int32_t vlen=0;
	bool vfree=false;
	float64_t* vec=NULL;

	while (changed && (iter<max_iter))
	{
		iter++;
		if (iter==max_iter-1)
			SG_SWARNING("kmeans clustering changed throughout %d iterations stopping...\n", max_iter-1)

		if (iter%1000 == 0)
			SG_SINFO("Iteration[%d/%d]: Assignment of %i patterns changed.\n", iter, max_iter, changed)
		changed=0;

		if (!fixed_centers)
		{
			/* mus=zeros(dimensions, k) ; */
			mus.zero();
			for (int32_t i=0; i<XSize; i++)
			{
				int32_t Cl=ClList[i];

				vec=lhs->get_feature_vector(i, vlen, vfree);

				for (int32_t j=0; j<dimensions; j++)
					mus.matrix[Cl*dimensions+j] += vec[j];

				lhs->free_feature_vector(vec, i, vfree);
			}

			for (int32_t i=0; i<k; i++)
			{
				if (weights_set[i]!=0.0)
				{
					for (int32_t j=0; j<dimensions; j++)
						mus.matrix[i*dimensions+j] /= weights_set[i];
				}
			}
		}
		rhs_mus->copy_feature_matrix(mus);
		for (int32_t i=0; i<XSize; i++)
		{
			/* ks=ceil(rand(1,XSize)*XSize) ; */
			const int32_t Pat=CMath::random(0, XSize-1);
			const int32_t ClList_Pat=ClList[Pat];
			int32_t imini, j;
			float64_t mini;

			/* compute the distance of this point to all centers */
			for(int32_t idx_k=0;idx_k<k;idx_k++)
				dists[idx_k]=distance->distance(Pat,idx_k);

			/* [mini,imini]=min(dists(:,i)) ; */
			imini=0 ; mini=dists[0];
			for (j=1; j<k; j++)
				if (dists[j]<mini)
				{
					mini=dists[j];
					imini=j;
				}

			if (imini!=ClList_Pat)
			{
				changed++;

				/* weights_set(imini) = weights_set(imini) + 1.0 ; */
				weights_set[imini]+= 1.0;
				/* weights_set(j)     = weights_set(j)     - 1.0 ; */
				weights_set[ClList_Pat]-= 1.0;

				vec=lhs->get_feature_vector(Pat, vlen, vfree);

				for (j=0; j<dimensions; j++)
				{
					mus.matrix[imini*dimensions+j]-=
						(vec[j]-mus.matrix[imini*dimensions+j]) / weights_set[imini];
				}

				lhs->free_feature_vector(vec, Pat, vfree);

				/* mu_new = mu_old - (x - mu_old)/(n-1) */
				/* if weights_set(j)~=0 */
				if (weights_set[ClList_Pat]!=0.0)
				{
					vec=lhs->get_feature_vector(Pat, vlen, vfree);
	
					for (j=0; j<dimensions; j++)
					{
						mus.matrix[ClList_Pat*dimensions+j]-=
								(vec[j]-mus.matrix[ClList_Pat*dimensions+j]) / weights_set[ClList_Pat];
					}
					lhs->free_feature_vector(vec, Pat, vfree);
				}
				else
				{
					/*  mus(:,j)=zeros(dimensions,1) ; */
					for (j=0; j<dimensions; j++)
						mus.matrix[ClList_Pat*dimensions+j]=0;
				}

				/* ClList(i)= imini ; */
				ClList[Pat] = imini;
			}
		}
	}
	distance->replace_rhs(rhs_cache);
	delete rhs_mus;
	SG_UNREF(lhs);
}
}
