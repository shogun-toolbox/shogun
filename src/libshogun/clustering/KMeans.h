/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2007-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KMEANS_H__
#define _KMEANS_H__

#include <stdio.h>
#include "lib/common.h"
#include "lib/io.h"
#include "features/SimpleFeatures.h"
#include "distance/Distance.h"
#include "machine/DistanceMachine.h"

namespace shogun
{
class CDistanceMachine;

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
 * cf. http://en.wikipedia.org/wiki/K-means_algorithm */
class CKMeans : public CDistanceMachine
{
	public:
		/** default constructor */
		CKMeans();

		/** constructor
		 *
		 * @param k parameter k
		 * @param d distance
		 */
		CKMeans(int32_t k, CDistance* d);
		virtual ~CKMeans();

		/** get classifier type
		 *
		 * @return classifier type KMEANS
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_KMEANS; }

		/** train k-means
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** load distance machine from file
		 *
		 * @param srcfile file to load from
		 * @return if loading was successful
		 */
		virtual bool load(FILE* srcfile);

		/** save distance machine to file
		 *
		 * @param dstfile file to save to
		 * @return if saving was successful
		 */
		virtual bool save(FILE* dstfile);

		/** set k
		 *
		 * @param p_k new k
		 */
		inline void set_k(int32_t p_k)
		{
			ASSERT(p_k>0);
			this->k=p_k;
		}

		/** get k
		 *
		 * @return the parameter k
		 */
		inline int32_t get_k()
		{
			return k;
		}

		/** set maximum number of iterations
		 *
		 * @param iter the new maximum
		 */
		inline void set_max_iter(int32_t iter)
		{
			ASSERT(iter>0);
			max_iter=iter;
		}

		/** get maximum number of iterations
		 *
		 * @return maximum number of iterations
		 */
		inline float64_t get_max_iter()
		{
			return max_iter;
		}

		/** get radi
		 *
		 */
		SGVector<float64_t> get_radiuses()
		{
			return SGVector<float64_t>(R,k);
		}

		/** get centers
		 *
		 */
		SGMatrix<float64_t> get_cluster_centers()
		{
			return SGMatrix<float64_t>(mus,dimensions,k);
		}

		/** get dimensions
		 *
		 * @return number of dimensions
		 */
		inline int32_t get_dimensions()
		{
			return dimensions;
		}

	protected:
		/** clustknb
		 *
		 * @param use_old_mus if old mus shall be used
		 * @param mus_start mus start
		 */
		void clustknb(bool use_old_mus, float64_t *mus_start);

		/** classify objects using the currently set features
		 *
		 * @return classified labels
		 */
		virtual CLabels* apply()
		{
			SG_NOTIMPLEMENTED;
			return NULL;
		}

		/** classify objects
		 *
		 * @param data (test)data to be classified
		 * @return classified labels
		 */
		virtual CLabels* apply(CFeatures* data)
		{
			SG_NOTIMPLEMENTED;
			return NULL;
		}



		/** @return object name */
		inline virtual const char* get_name() const { return "KMeans"; }

	protected:
		/// maximum number of iterations
		int32_t max_iter;

		/// the k parameter in KMeans
		int32_t k;

		/// number of dimensions
		int32_t dimensions;

		/// radi of the clusters (size k)
		float64_t* R;
		
		/// centers of the clusters (size dimensions x k)
		float64_t* mus;

	private:
		/// weighting over the train data
		float64_t* Weights;
};
}
#endif

