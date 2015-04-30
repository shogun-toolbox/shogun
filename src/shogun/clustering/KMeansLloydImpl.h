/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Parijat Mazumdar
 */

#ifndef _LKMEANS_H__
#define _LKMEANS_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>

namespace shogun
{
/** Implementation class for the standard KMeans algorithm. */
class CKMeansLloydImpl
{
	public:
		/** Lloyd's KMeans training method
		 *
		 * @param k parameter k
		 * @param distance distance
		 * @param max_iter max iterations allowed
		 * @param mus cluster centers matrix (k columns)
		 * @param ClList cluster number each data vector belongs (size no_of_vectors)
		 * @param weights_set no. of points belonging to each cluster (size k)
		 * @param fixed_centers keep centers fixed or not
		 */
		static void Lloyd_KMeans(int32_t k, CDistance* distance, int32_t max_iter, SGMatrix<float64_t> mus,
			SGVector<int32_t> ClList, SGVector<float64_t> weights_set, bool fixed_centers);
};
}
#endif
