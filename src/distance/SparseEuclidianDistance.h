/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2008 Soeren Sonnenburg
 * Copyright (C) 2007-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPARSEEUCLIDIANDISTANCE_H__
#define _SPARSEEUCLIDIANDISTANCE_H__

#include "lib/common.h"
#include "distance/SparseDistance.h"
#include "features/SparseFeatures.h"

class CSparseEuclidianDistance: public CSparseDistance<DREAL>
{
	public:
		CSparseEuclidianDistance();
		CSparseEuclidianDistance(CSparseFeatures<DREAL>* l, CSparseFeatures<DREAL>* r);
		virtual ~CSparseEuclidianDistance();

		virtual bool init(CFeatures* l, CFeatures* r);
		virtual void cleanup();

		/// load and save kernel init_data
		virtual bool load_init(FILE* src);
		virtual bool save_init(FILE* dest);

		// return what type of kernel we are Linear,Polynomial, Gaussian,...
		virtual EDistanceType get_distance_type() { return D_SPARSEEUCLIDIAN; }

		/** return feature type the kernel can deal with
		*/
		inline virtual EFeatureType get_feature_type() { return F_DREAL; }

		// return the name of a kernel
		virtual const CHAR* get_name() { return "SparseEuclidian" ; } ;

	protected:
		/// compute kernel function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual DREAL compute(INT idx_a, INT idx_b);
		/*    compute_kernel*/

	protected:
		double scale;
		DREAL* sq_lhs;
		DREAL* sq_rhs;

};

#endif /* _SPARSEEUCLIDIANDISTANCE_H__ */
