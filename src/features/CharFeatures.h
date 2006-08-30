/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CHARFEATURES__H__
#define _CHARFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

class CCharFeatures: public CSimpleFeatures<CHAR>
{
	public:
		CCharFeatures(E_ALPHABET alphabet, INT size);
		CCharFeatures(const CCharFeatures & orig);
        CCharFeatures(E_ALPHABET alphabet, CHAR* feature_matrix, INT num_feat, INT num_vec);
		CCharFeatures(E_ALPHABET alphabet, CHAR* fname);

		/// remap element e.g translate ACGT to 0123
		inline BYTE remap(BYTE c)
		{
			return maptable[c];
		}

		inline E_ALPHABET get_alphabet()
		{
			return alphabet_type;
		}

		virtual EFeatureType get_feature_type() { return F_CHAR; }

		virtual CFeatures* duplicate() const;
		virtual bool load(CHAR* fname);
		virtual bool save(CHAR* fname);
	public:
		static const BYTE B_A;
		static const BYTE B_C;
		static const BYTE B_G;
		static const BYTE B_T;
		static const BYTE B_star;
		static const BYTE B_N;
		static const BYTE B_n;
	protected:
		void init_map_table();
		BYTE maptable[1 << (sizeof(CHAR)*8)];
		E_ALPHABET alphabet_type;
};
#endif
