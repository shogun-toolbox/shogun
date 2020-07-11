/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Saurabh Mahindre, 
 *          Sergey Lisitsyn, Bjoern Esser, Evan Shelhamer, Yuyu Zhang, 
 *          Fernando Iglesias, parijat
 */

#ifndef _KMEANS_H__
#define _KMEANS_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/machine/DistanceMachine.h>
#include <shogun/clustering/KMeansBase.h>

namespace shogun
{
class KMeansBase;

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
 * To use mini-batch based training was see KMeansMiniBatch 
 *
 * cf. http://en.wikipedia.org/wiki/K-means_algorithm
 * cf. http://en.wikipedia.org/wiki/Lloyd's_algorithm
 *
 *
 */
class KMeans : public KMeansBase
{
	public:
	
		/** default constructor */
		KMeans();

		/** constructor
		 *
		 * @param k parameter k
		 * @param d distance
		 * @param kmeanspp Set to true for using KMeans++ (default false)
		 */
		KMeans(int32_t k, std::shared_ptr<Distance> d, bool kmeanspp=false);

		/** constructor for supplying initial centers
		 * @param k_i parameter k
		 * @param d_i distance
		 * @param centers_i initial centers for KMeans algorithm
		 */
		KMeans(int32_t k_i, std::shared_ptr<Distance> d_i, SGMatrix<float64_t> centers_i);

		~KMeans() override;

		/** @return object name */
		const char* get_name() const override { return "KMeans"; }		

	private:

		/** train k-means
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		bool train_machine(std::shared_ptr<Features> data=NULL) override;

		/** Lloyd's KMeans training method
		 */
		void Lloyd_KMeans(SGMatrix<float64_t> centers, int32_t num_centers);
};
}
#endif
