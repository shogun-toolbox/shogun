/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _INTFEATURES__H__
#define _INTFEATURES__H__

#include "features/SimpleFeatures.h"
#include "lib/common.h"

/** The class IntFeatures implements a dense integer matrix.
 * It inherits its functionality from CSimpleFeatures, which should be
 * consulted for further reference.
 */
class CIntFeatures: public CSimpleFeatures<int32_t>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		CIntFeatures(int32_t size=0) : CSimpleFeatures<int32_t>(size) {}

		/** copy constructor */
		CIntFeatures(const CIntFeatures & orig) :
			CSimpleFeatures<int32_t>(orig) {}

        /** constructor that copies feature matrix from
         * pointer num_feat,num_vec pair
		 *
		 * @param src feature matrix to copy
		 * @param num_feat number of features
		 * @param num_vec number of vectors
		 */
		inline CIntFeatures(int32_t* src, int32_t num_feat, int32_t num_vec): CSimpleFeatures<int32_t>(0)
		{
			CSimpleFeatures<int32_t>::copy_feature_matrix(src, num_feat, num_vec);
		}

		/** constructor
		 *
		 * @param fname filename to load features from
		 */
		CIntFeatures(char* fname) : CSimpleFeatures<int32_t>(fname)
		{
			load(fname);
		}

		/** get feature matrix
		 *
		 * @param dst destination where matrix will be stored
		 * @param d1 dimension 1 of matrix
		 * @param d2 dimension 2 of matrix
		 */
		inline virtual void get_fm(int32_t** dst, int32_t* d1, int32_t* d2)
		{
			CSimpleFeatures<int32_t>::get_fm(dst, d1, d2);
		}

		/** copy feature matrix
		 *
		 * wrapper to base class' method
		 *
		 * @param src feature matrix to copy
		 * @param num_feat number of features
		 * @param num_vec number of vectors
		 */
		inline virtual void copy_feature_matrix(int32_t* src, int32_t num_feat, int32_t num_vec)
		{
			CSimpleFeatures<int32_t>::copy_feature_matrix(src, num_feat, num_vec);
		}

		/** load features from file
		 *
		 * @param fname filename to load from
		 * @return if loading was successful
		 */
		virtual bool load(char* fname);

		/** save features to file
		 *
		 * @param fname filename to save to
		 * @return if saving was successful
		 */
		virtual bool save(char* fname);
};
#endif
