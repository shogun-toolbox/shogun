/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _EUCLIDIANDISTANCE_H__
#define _EUCLIDIANDISTANCE_H__

#include "lib/common.h"
#include "distance/RealDistance.h"
#include "features/RealFeatures.h"

class CEuclidianDistance: public CRealDistance
{
	public:
		CEuclidianDistance();
		CEuclidianDistance(CRealFeatures* l, CRealFeatures* r);
		virtual ~CEuclidianDistance();

		virtual bool init(CFeatures* l, CFeatures* r);
		virtual void cleanup();

		/// load and save kernel init_data
		virtual bool load_init(FILE* src);
		virtual bool save_init(FILE* dest);

		// return what type of kernel we are Linear,Polynomial, Gaussian,...
		virtual EDistanceType get_distance_type() { return D_EUCLIDIAN; }

		/** return feature type the kernel can deal with
		*/
		inline virtual EFeatureType get_feature_type() { return F_DREAL; }

		// return the name of a kernel
		virtual const CHAR* get_name() { return "Euclidian" ; } ;

		/*
		 * disable application of sqrt on matrix computation
		 * the matrix can then also be named norm squared
		 */
		virtual bool get_disable_sqrt() { return disable_sqrt; };
		virtual void set_disable_sqrt(bool state) { disable_sqrt=state; };

	protected:
		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual DREAL compute(INT idx_a, INT idx_b);
		/*    compute_kernel*/

	protected:
		double scale;
		bool disable_sqrt;
};

#endif /* _EUCLIDIANDISTANCE_H__ */
