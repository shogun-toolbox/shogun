/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SHORTREALFEATURES__H__
#define _SHORTREALFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

/** class RealFeatures */
class CRealFeatures: public CSimpleFeatures<SHORTREAL>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		CRealFeatures(INT size=0) : CSimpleFeatures<SHORTREAL>(size) {}

		/** copy constructor */
		CRealFeatures(const CRealFeatures & orig) :
			CSimpleFeatures<SHORTREAL>(orig) {}

		/** constructor
		 *
		 * @param fname filename to load features from
		 */
		CRealFeatures(CHAR* fname) : CSimpleFeatures<SHORTREAL>(fname)
		{
			load(fname);
		}

		/** get feature matrix
		 *
		 * @param dst destination where matrix will be stored
		 * @param d1 dimension 1 of matrix
		 * @param d2 dimension 2 of matrix
		 */
		inline virtual void get_fm(SHORTREAL** dst, INT* d1, INT* d2)
		{
			CSimpleFeatures<SHORTREAL>::get_fm(dst, d1, d2);
		}

		/** copy feature matrix
		 *
		 * wrapper to base class' method
		 *
		 * @param src feature matrix to copy
		 * @param num_feat number of features
		 * @param num_vec number of vectors
		 */
		inline virtual void copy_feature_matrix(SHORTREAL* src, INT num_feat, INT num_vec)
		{
			CSimpleFeatures<SHORTREAL>::copy_feature_matrix(src, num_feat, num_vec);
		}

		/** load features from file
		 *
		 * @param fname filename to load from
		 * @return if loading was successful
		 */
		virtual bool load(CHAR* fname);

		/** save features to file
		 *
		 * @param fname filename to save to
		 * @return if saving was successful
		 */
		virtual bool save(CHAR* fname);
};
#endif
