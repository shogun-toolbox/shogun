/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Parijat Mazumdar
 */

#ifndef _KMEANS_H__
#define _KMEANS_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/clustering/KMeansBase.h>

namespace shogun
{
class CKMeansBase;
/** @brief KMeans clustering,  partitions the data into k (a-priori specified) clusters.
 *
 * It minimizes
 * \f[
 *  \sum_{i=1}^k\sum_{x_j\in S_i} (x_j-\mu_i)^2
 * \f]
 *
 * where \f$\mu_i\f$ are the cluster centers and \f$S_i,\;i=1,\dots,k\f$ are the index
 * sets of the clusters.
 *
 * Beware that this algorithm obtains only a <em>local</em> optimum.
 *
 * This class uses the Lloyd's algorithm.
 * 
 * Mini-batch based training can be done using the CKMeansMiniBatch class.
 *
 * cf. http://en.wikipedia.org/wiki/K-means_algorithm
 * cf. http://en.wikipedia.org/wiki/Lloyd's_algorithm
 *
 *
 */
class CKMeans : public CKMeansBase
{
	public:
	
		/** default constructor */
		CKMeans();

		/** constructor
		 *
		 * @param k parameter k
		 * @param d distance
		 * @param kmeanspp true for using KMeans++ (default false)
		 */
		CKMeans(int32_t k, CDistance* d, bool kmeanspp=false);

		/** constructor for supplying initial centers
		 * @param k_i parameter k
		 * @param d_i distance
		 * @param centers_i initial centers for KMeans algorithm
		*/
		CKMeans(int32_t k_i, CDistance* d_i, SGMatrix<float64_t> centers_i);
		virtual ~CKMeans();

		/** @return object name */
		virtual const char* get_name() const { return "KMeans"; }		

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

};
}
#endif
