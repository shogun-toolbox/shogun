/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST
 */

#ifndef _CANBERRAMETRIC_H__
#define _CANBERRAMETRIC_H___

#include "lib/common.h"
#include "distance/SimpleDistance.h"
#include "features/RealFeatures.h"

class CCanberraMetric: public CSimpleDistance<DREAL>
{
	public:
		CCanberraMetric();
		CCanberraMetric(CRealFeatures* l, CRealFeatures* r);
		virtual ~CCanberraMetric();

		virtual bool init(CFeatures* l, CFeatures* r);
		virtual void cleanup();

		/// load and save distance init_data
		virtual bool load_init(FILE* src);
		virtual bool save_init(FILE* dest);

		// return type of distance
		virtual EDistanceType get_distance_type() { return D_CANBERRA; }

		// return the name of a distance
		virtual const CHAR* get_name() { return "Canberra-Metric"; };

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual DREAL compute(INT idx_a, INT idx_b);
};

#endif /* _CANBERRAMETRIC_H__ */
