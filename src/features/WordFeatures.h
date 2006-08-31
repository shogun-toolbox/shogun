/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WORDFEATURES__H__
#define _WORDFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

class CWordFeatures: public CSimpleFeatures<WORD>
{
	public:
		CWordFeatures(INT size, INT num_symbols=1<<16);
		CWordFeatures(const CWordFeatures & orig);

		/** load features from file
		 * fname - filename
		 */

		CWordFeatures(CHAR* fname, INT num_symbols=1<<16);

		virtual ~CWordFeatures();

		bool obtain_from_char_features(CCharFeatures* cf, INT start, INT order, INT gap=0);

		virtual EFeatureType get_feature_type() { return F_WORD; }

		virtual CFeatures* duplicate() const;
		virtual bool load(CHAR* fname);
		virtual bool save(CHAR* fname);

		inline INT get_num_symbols() { return num_symbols; }

		// these functions are necessary to find out about a former conversion process

		// number of symbols before higher order mapping
		inline INT get_original_num_symbols() { return original_num_symbols; }

		// order used for higher order mapping
		inline INT get_order() { return order; }

		// a higher order mapped symbol will be shaped such that the symbols in
		// specified by bits in the mask will be returned.
		inline WORD get_masked_symbols(WORD symbol, BYTE mask)
		{
			ASSERT(symbol_mask_table);
			return symbol_mask_table[mask] & symbol;
		}

	protected:
		///max_val is how many bits does the largest symbol require to be stored without loss
		void translate_from_single_order(WORD* obs, INT sequence_length, INT start, INT order, INT max_val, INT gap=0);

	protected:
		/// number of used symbols
		INT num_symbols;

		/// original number of used symbols (before higher order mapping)
		INT original_num_symbols;

		/// order used in higher order mapping
		INT order;

		/// order used in higher order mapping
		WORD* symbol_mask_table;
};
#endif
