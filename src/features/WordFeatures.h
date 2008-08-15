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

#ifndef _WORDFEATURES__H__
#define _WORDFEATURES__H__

#include "features/SimpleFeatures.h"
#include "features/CharFeatures.h"
#include "lib/common.h"

/** The class WordFeatures implements a dense word (16bit unsigned) matrix.
 * It inherits its functionality from CSimpleFeatures, which should be
 * consulted for further reference.
 */
class CWordFeatures : public CSimpleFeatures<WORD>
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param num_symbols number of symbols
		 */
		CWordFeatures(INT size=0, INT num_symbols=(1<<16));

		/** copy constructor */
		CWordFeatures(const CWordFeatures & orig);

        /** constructor that copies feature matrix from
         * pointer num_feat,num_vec pair
		 *
		 * @param src feature matrix to copy
		 * @param num_feat number of features
		 * @param num_vec number of vectors
		 */
		inline CWordFeatures(WORD* src, INT num_feat, INT num_vec):
            CSimpleFeatures<WORD>(0), num_symbols(1<<16),
			original_num_symbols(1<<16), order(0), symbol_mask_table(NULL)
		{
			CSimpleFeatures<WORD>::copy_feature_matrix(src, num_feat, num_vec);
		}

		/** constructor
		 *
		 * @param fname filename to load features from
		 * @param num_sym number of symbols
		 */
		CWordFeatures(CHAR* fname, INT num_sym = (1<<16));

		virtual ~CWordFeatures();

		/** obtain from char features
		 *
		 * @param cf char features
		 * @param start start
		 * @param order order
		 * @param gap gap
		 * @return if obtaining was successful
		 */
		bool obtain_from_char_features(CCharFeatures* cf, INT start, INT order, INT gap=0);

		/** copy feature matrix
		 *
		 * wrapper to base class' method
		 *
		 * @param src feature matrix to copy
		 * @param num_feat number of features
		 * @param num_vec number of vectors
		 */
		inline virtual void copy_feature_matrix(WORD* src, INT num_feat, INT num_vec)
		{
			CSimpleFeatures<WORD>::copy_feature_matrix(src, num_feat, num_vec);
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

		/** get number of symbols
		 *
		 * @return number of symbols
		 */
		inline INT get_num_symbols() { return num_symbols; }

		// these functions are necessary to find out about a former conversion process

		/** number of symbols before higher order mapping
		 *
		 * @return original number of symbols
		 */
		inline INT get_original_num_symbols() { return original_num_symbols; }

		/** order used for higher order mapping
		 *
		 * @return order
		 */
		inline INT get_order() { return order; }

		/** a higher order mapped symbol will be shaped such that the symbols in
		 * specified by bits in the mask will be returned.
		 *
		 * @param symbol symbol to mask
		 * @param mask mask to apply
		 * @return masked symbol
		 */
		inline WORD get_masked_symbols(WORD symbol, BYTE mask)
		{
			ASSERT(symbol_mask_table);
			return symbol_mask_table[mask] & symbol;
		}

	protected:
		/** translate from single order
		 *
		 * @param obs observation
		 * @param sequence_length length of sequence
		 * @param start start
		 * @param order order
		 * @param max_val how many bits does the largest symbol
		 *                require to be stored without loss
		 * @param gap gap
		 */
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
