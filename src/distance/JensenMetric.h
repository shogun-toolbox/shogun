/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _JENSENMETRIC_H___
#define _JENSENMETRIC_H___

#include "lib/common.h"
#include "distance/SimpleDistance.h"
#include "features/RealFeatures.h"

class CJensenMetric: public CSimpleDistance<DREAL>
{
	public:
		CJensenMetric();
		CJensenMetric(CRealFeatures* l, CRealFeatures* r);
		virtual ~CJensenMetric();

		virtual bool init(CFeatures* l, CFeatures* r);
		virtual void cleanup();

		/// load and save distance init_data
		virtual bool load_init(FILE* src);
		virtual bool save_init(FILE* dest);

		// return type of distance
		virtual EDistanceType get_distance_type() { return D_JENSEN; }

		// return the name of distance
		virtual const CHAR* get_name() { return "Jensen-Metric"; };

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual DREAL compute(INT idx_a, INT idx_b);
};

#endif /* _JENSENMETRIC_H___ */
