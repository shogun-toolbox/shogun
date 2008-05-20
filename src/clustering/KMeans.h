/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2007-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _KMEANS_H__
#define _KMEANS_H__

#include <stdio.h>
#include "lib/common.h"
#include "lib/io.h"
#include "features/RealFeatures.h"
#include "distance/Distance.h"
#include "distance/DistanceMachine.h"

class CDistanceMachine;

/** class KMeans */
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
		CKMeans(INT k, CDistance* d);
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
		inline void set_k(INT p_k)
		{
			ASSERT(p_k>0);
			this->k=p_k;
		}

		/** get k
		 *
		 * @return the parameter k
		 */
		inline INT get_k()
		{
			return k;
		}

		/** set maximum number of iterations
		 *
		 * @param iter the new maximum
		 */
		inline void set_max_iter(INT iter)
		{
			ASSERT(iter>0);
			max_iter=iter;
		}

		/** get maximum number of iterations
		 *
		 * @return maximum number of iterations
		 */
		inline DREAL get_max_iter()
		{
			return max_iter;
		}

		/** get radi
		 *
		 * @param radi current radi are stored in here
		 * @param num number of radi is stored in here
		 */
		inline void get_radi(DREAL*& radi, INT& num)
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
		inline void get_centers(DREAL*& centers, INT& dim, INT& num)
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
		inline void get_radiuses(DREAL** radii, INT* num)
		{
			size_t sz=sizeof(*R)*k;
			*radii=(DREAL*) malloc(sz);
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
		inline void get_cluster_centers(DREAL** centers, INT* dim, INT* num)
		{
			size_t sz=sizeof(*mus)*dimensions*k;
			*centers=(DREAL*) malloc(sz);
			ASSERT(*centers);

			memcpy(*centers, mus, sz);
			*dim=dimensions;
			*num=k;
		}

		/** get dimensions
		 *
		 * @return number of dimensions
		 */
		inline INT get_dimensions()
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
		void sqdist(double * x, CRealFeatures* y, double *z,
				int n1, int offs, int n2, int m);

		/** clustknb
		 *
		 * @param use_old_mus if old mus shall be used
		 * @param mus_start mus start
		 */
		void clustknb(bool use_old_mus, double *mus_start);

	protected:
		/// maximum number of iterations
		INT max_iter;

		/// the k parameter in KMeans
		INT k;

		/// number of dimensions
		INT dimensions;

		/// radi of the clusters (size k)
		DREAL* R;
		
		/// centers of the clusters (size dimensions x k)
		DREAL* mus;

	private:
		/// weighting over the train data
		DREAL* Weights;
};
#endif

