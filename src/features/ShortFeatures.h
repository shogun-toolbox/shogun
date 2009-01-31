/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SHORTFEATURES__H__
#define _SHORTFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

/** The class ShortFeatures implements a dense short integer matrix.
 * It inherits its functionality from CSimpleFeatures, which should be
 * consulted for further reference.
 */
class CShortFeatures : public CSimpleFeatures<int16_t>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		CShortFeatures(int32_t size=0);

		/** copy constructor */
		CShortFeatures(const CShortFeatures & orig);

        /** constructor that copies feature matrix from
         * pointer num_feat,num_vec pair
		 *
		 * @param src feature matrix to copy
		 * @param num_feat number of features
		 * @param num_vec number of vectors
		 */
		inline CShortFeatures(int16_t* src, int32_t num_feat, int32_t num_vec): CSimpleFeatures<int16_t>(0)
		{
			CSimpleFeatures<int16_t>::copy_feature_matrix(src, num_feat, num_vec);
		}

		/** constructor
		 *
		 * @param fname filename to load features from
		 */
		CShortFeatures(char* fname);

		/** obtain from char features
		 *
		 * @param cf char features
		 * @param start start
		 * @param order order
		 * @param gap gap
		 * @return if obtaining was successful
		 */
		bool obtain_from_char_features(CCharFeatures* cf, int32_t start, int32_t order, int32_t gap=0);

		/** get feature type
		 *
		 * @return feature type SHORT
		 */
		virtual EFeatureType get_feature_type() { return F_SHORT; }

		/** copy feature matrix
		 *
		 * wrapper to base class' method
		 *
		 * @param src feature matrix to copy
		 * @param num_feat number of features
		 * @param num_vec number of vectors
		 */
		inline virtual void copy_feature_matrix(int16_t* src, int32_t num_feat, int32_t num_vec)
		{
			CSimpleFeatures<int16_t>::copy_feature_matrix(src, num_feat, num_vec);
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
	protected:
		/** translate from single order
		 *
		 * @param obs observation
		 * @param sequence_length length of sequence
		 * @param start start
		 * @param order order
		 * @param max_val maximum value
		 * @param gap gap
		 */
		void translate_from_single_order(int16_t* obs, int32_t sequence_length, int32_t start, int32_t order, int32_t max_val, int32_t gap);

		/** @return object name */
		inline virtual const char* get_name() { return "ShortFeatures"; }
};
#endif
