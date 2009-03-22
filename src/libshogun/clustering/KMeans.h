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
#include "distance/DistanceMachine.h"

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

		/** train distance machine
		 *
		 * @return if training was successful
		 */
		virtual bool train();

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
		 * @param radi current radi are stored in here
		 * @param num number of radi is stored in here
		 */
		inline void get_radi(float64_t*& radi, int32_t& num)
		{
			radi=R;
			num=k;
		}

		/** get centers
		 *
		 * @param centers current centers are stored in here
		 * @param dim dimensions are stored in here
		 * @param num number of centers is stored in here
		 */
		inline void get_centers(float64_t*& centers, int32_t& dim, int32_t& num)
		{
			centers=mus;
			dim=dimensions;
			num=k;
		}

		/** get radiuses (swig compatible)
		 *
		 * @param radii current radiuses are stored in here
		 * @param num number of radiuses is stored in here
		 */
		inline void get_radiuses(float64_t** radii, int32_t* num)
		{
			size_t sz=sizeof(*R)*k;
			*radii=(float64_t*) malloc(sz);
			ASSERT(*radii);

			memcpy(*radii, R, sz);
			*num=k;
		}

		/** get cluster centers (swig compatible)
		 *
		 * @param centers current cluster centers are stored in here
		 * @param dim dimensions are stored in here
		 * @param num number of centers is stored in here
		 */
		inline void get_cluster_centers(
			float64_t** centers, int32_t* dim, int32_t* num)
		{
			size_t sz=sizeof(*mus)*dimensions*k;
			*centers=(float64_t*) malloc(sz);
			ASSERT(*centers);

			memcpy(*centers, mus, sz);
			*dim=dimensions;
			*num=k;
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
		/** sqdist
		 *
		 * @param x x
		 * @param y y
		 * @param z z
		 * @param n1 n1
		 * @param offs offset
		 * @param n2 n2
		 * @param m m
		 */
		void sqdist(
			float64_t* x, CSimpleFeatures<float64_t>* y, float64_t *z, int32_t n1,
			int32_t offs, int32_t n2, int32_t m);

		/** clustknb
		 *
		 * @param use_old_mus if old mus shall be used
		 * @param mus_start mus start
		 */
		void clustknb(bool use_old_mus, float64_t *mus_start);

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
#endif

