/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CHARFEATURES__H__
#define _CHARFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/Alphabet.h"
#include "lib/common.h"

/** class CharFeatures */
class CCharFeatures : public CSimpleFeatures<CHAR>
{
	public:
		/** constructor
		 *
		 * @param alpha alphabet (type) to use
		 * @param size cache size
		 */
		CCharFeatures(E_ALPHABET alpha, INT size=0);

		/** constructor
		 *
		 * @param alpha alphabet to use
		 * @param size cache size
		 */
		CCharFeatures(CAlphabet* alpha, INT size=0);

		/** copy constructor */
		CCharFeatures(const CCharFeatures & orig);

		/** constructor
		 *
		 * @param alphabet alphabet to use
		 * @param feature_matrix feature matrix to use
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 */
		CCharFeatures(E_ALPHABET alphabet, CHAR* feature_matrix, INT num_feat, INT num_vec);

		/** constructor
		 *
		 * @param alphabet alphabet (type) to use
		 * @param fname filename to load features from
		 */
		CCharFeatures(E_ALPHABET alphabet, CHAR* fname);

		~CCharFeatures();

		/** get alphabet
		 *
		 * @return alphabet
		 */
		inline CAlphabet* get_alphabet()
		{
			return alphabet;
		}

		/** copy feature matrix
		 *
		 * wrapper to base class' method
		 *
		 * @param src feature matrix to copy
		 * @param num_feat number of features
		 * @param num_vec number of vectors
		 */
		inline virtual void copy_feature_matrix(CHAR* src, INT num_feat, INT num_vec)
		{
			CSimpleFeatures<CHAR>::copy_feature_matrix(src, num_feat, num_vec);
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

	protected:
		/** alphabet */
		CAlphabet* alphabet;
};
#endif
