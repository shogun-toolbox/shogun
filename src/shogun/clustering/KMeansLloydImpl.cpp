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
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

namespace shogun
{
void CKMeansLloydImpl::Lloyd_KMeans(int32_t k, CDistance* distance, int32_t max_iter, SGMatrix<float64_t> mus,
		SGVector<int32_t> ClList, SGVector<float64_t> weights_set, bool fixed_centers)
{
	CDenseFeatures<float64_t>* lhs=
		CDenseFeatures<float64_t>::obtain_from_generic(distance->get_lhs());
	int32_t XSize=lhs->get_num_vectors();
	int32_t dimensions=lhs->get_num_features();

	CFeatures* rhs_cache=distance->replace_rhs(mus);

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
		for (int32_t i=0; i<XSize; i++)
		{
			const int32_t ClList_i=ClList[i];
			int32_t imini, j;
			float64_t mini;

			/* compute the distance of this point to all centers */
			for(int32_t idx_k=0;idx_k<k;idx_k++)
				dists[idx_k]=distance->distance(i,idx_k);

			/* [mini,imini]=min(dists(:,i)) ; */
			imini=0 ; mini=dists[0];
			for (j=1; j<k; j++)
				if (dists[j]<mini)
				{
					mini=dists[j];
					imini=j;
				}

			if (imini!=ClList_i)
			{
				changed++;
				ClList[i] = imini;
			}
		}
		if (fixed_centers)
			break;
	}
	distance->replace_rhs(rhs_cache);
	SG_UNREF(lhs);
}
}
