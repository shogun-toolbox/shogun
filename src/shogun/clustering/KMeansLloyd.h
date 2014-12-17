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
#include <shogun/clustering/KMeans.h>

namespace shogun
{
class CKMeans;
/** Implementation class for the standard KMeans algorithm. */
class CKMeansLloyd : public CKMeans
{
	public:
	
		/** default constructor */
		CKMeansLloyd();

		/** constructor
		 *
		 * @param k parameter k
		 * @param d distance
		 * @param kmeanspp true for using KMeans++ (default false)
		 */
		CKMeansLloyd(int32_t k, CDistance* d, bool kmeanspp=false);

		/** constructor for supplying initial centers
		 * @param k_i parameter k
		 * @param d_i distance
		 * @param centers_i initial centers for KMeans algorithm
		*/
		CKMeansLloyd(int32_t k_i, CDistance* d_i, SGMatrix<float64_t> centers_i);
		virtual ~CKMeansLloyd();



	private:

		/** train k-means
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train_machine(CFeatures* data=NULL);

	
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
		void Lloyd_KMeans(SGVector<int32_t> ClList, SGVector<float64_t> weights_set);
};
}
#endif
