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

class CKMeans : public CDistanceMachine
{
	public:
		CKMeans();
		CKMeans(INT k, CDistance* d);
		virtual ~CKMeans();

		virtual inline EClassifierType get_classifier_type() { return CT_KMEANS; }

		virtual bool train();

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);

		inline void set_k(INT p_k) 
		{
			ASSERT(p_k>0);
			this->k=p_k;
		}

		inline DREAL get_k()
		{
			return k;
		}

		inline void set_max_iter(INT iter) 
		{
			ASSERT(iter>0);
			max_iter=iter;
		}

		inline DREAL get_max_iter()
		{
			return max_iter;
		}

		inline void get_radi(DREAL*& radi, INT& num)
		{
			radi=R;
			num=k;
		}

		inline void get_centers(DREAL*& centers, INT& dim, INT& num)
		{
			centers=mus;
			dim=dimensions;
			num=k;
		}

		inline void get_radi(DREAL** radi, INT* num)
		{
			size_t sz=sizeof(*R)*k;
			*radi= (DREAL*) malloc(sz);
			ASSERT(*radi);

			memcpy(*radi, R, sz);
			*num=k;
		}

		inline INT get_dimensions()
		{
			return dimensions;
		}

		inline void get_centers(DREAL** centers, INT* dim, INT* num)
		{
			size_t sz=sizeof(*mus)*dimensions*k;
			*centers= (DREAL*) malloc(sz);
			ASSERT(*centers);

			memcpy(*centers, mus, sz);
			*dim=dimensions;
			*num=k;
		}


	protected:
		void sqdist(double * x, CRealFeatures* y, double *z,
				int n1, int offs, int n2, int m);

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

		/// weighting over the train data
		DREAL* Weights;
};
#endif

