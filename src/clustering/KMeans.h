/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Gunnar Raetsch
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
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
		virtual ~CKMeans();

		virtual inline EClassifierType get_classifier_type() { return CT_KMEANS; }

		virtual bool train();
		virtual CLabels* classify(CLabels* output=NULL);
		virtual DREAL classify_example(INT idx)
		{
			SG_ERROR( "for performance reasons use classify() instead of classify_example\n");
			return 0;
		}

		virtual bool load(FILE* srcfile);
		virtual bool save(FILE* dstfile);

		inline void set_k(DREAL p_k) 
		{
			ASSERT(k>0);
			this->k=p_k;
		}

		inline DREAL get_k()
		{
			return k;
		}

	protected:
		void sqdist(double * x, CRealFeatures* y, double *z,
				int n1, int offs, int n2, int m);

		void clustknb(bool use_old_mus, double *mus_start, bool disp);

	protected:
		/// the k parameter in KMeans
		INT k;

		/// radi of the clusters
		DREAL* R;
		
		/// centers of the clusters
		DREAL* mus;

		/// weighting over the train data
		DREAL* Weights;
};
#endif

