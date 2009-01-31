/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _BYTEFEATURES__H__
#define _BYTEFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "features/Alphabet.h"
#include "lib/common.h"

/** The class class ByteFeatures implements a dense byte matrix.
 * It inherits its functionality from CSimpleFeatures, which should be
 * consulted for further reference.
 */
class CByteFeatures : public CSimpleFeatures<uint8_t>
{
	public:
		/** constructor
		 *
		 * @param alpha alphabet (type) to use
		 * @param size cache size
		 */
		CByteFeatures(EAlphabet alpha, int32_t size=0);

		/** constructor
		 *
		 * @param alpha alphabet to use
		 * @param size cache size
		 */
		CByteFeatures(CAlphabet* alpha, int32_t size=0);

		/** copy constructor */
		CByteFeatures(const CByteFeatures & orig);
		
		/** constructor
		 *
		 * @param alphabet alphabet to use
		 * @param feature_matrix feature matrix to use
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 */
        CByteFeatures(EAlphabet alphabet, uint8_t* feature_matrix, int32_t num_feat, int32_t num_vec);

		/** constructor
		 *
		 * @param alphabet alphabet (type) to use
		 * @param fname filename to load features from
		 */
		CByteFeatures(EAlphabet alphabet, char* fname);

		~CByteFeatures();

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
		inline virtual void copy_feature_matrix(uint8_t* src, int32_t num_feat, int32_t num_vec)
		{
			CSimpleFeatures<uint8_t>::copy_feature_matrix(src, num_feat, num_vec);
		}

		/** get feature type
		 *
		 * @return feature type BYTE
		 */
		virtual EFeatureType get_feature_type() { return F_BYTE; }

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

		/** @return object name */
		inline virtual const char* get_name() { return "ByteFeatures"; }

	protected:
		/** alphabet */
		CAlphabet* alphabet;
};
#endif
