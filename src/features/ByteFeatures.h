/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _BYTEFEATURES__H__
#define _BYTEFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "features/Alphabet.h"
#include "lib/common.h"

class CByteFeatures: public CSimpleFeatures<BYTE>
{
	public:
		CByteFeatures(E_ALPHABET, INT size);
		CByteFeatures(CAlphabet* alpha, INT size);
		CByteFeatures(const CByteFeatures & orig);
        CByteFeatures(E_ALPHABET alphabet, BYTE* feature_matrix, INT num_feat, INT num_vec);
		CByteFeatures(E_ALPHABET alphabet, CHAR* fname);

		~CByteFeatures();

		inline CAlphabet* get_alphabet()
		{
			return alphabet;
		}

		inline virtual void copy_feature_matrix(BYTE* src, INT num_feat, INT num_vec)
		{
			CSimpleFeatures<BYTE>::copy_feature_matrix(src, num_feat, num_vec);
		}

		virtual EFeatureType get_feature_type() { return F_BYTE; }

		virtual bool load(CHAR* fname);
		virtual bool save(CHAR* fname);
	protected:
		CAlphabet* alphabet;
};
#endif
