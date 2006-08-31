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

#ifndef __GUIFEATURES__H
#define __GUIFEATURES__H

#include "features/Labels.h"
#include "features/Features.h"
#include "features/RealFileFeatures.h"
#include "features/TOPFeatures.h"
#include "features/FKFeatures.h"
#include "features/CharFeatures.h"
#include "features/StringFeatures.h"
#include "features/ByteFeatures.h"
#include "features/WordFeatures.h"
#include "features/ShortFeatures.h"
#include "features/RealFeatures.h"
#include "features/SparseRealFeatures.h"
#include "features/CombinedFeatures.h"
#include "features/MindyGramFeatures.h"

class CGUI;

class CGUIFeatures
{
	enum EFeatureType
	{
		Simple,
		Sparse
	};

	public:
		CGUIFeatures(CGUI *);
		~CGUIFeatures();

		inline CFeatures *get_train_features() { return train_features; }
		inline CFeatures *get_test_features() { return test_features; }

		inline bool set_train_features(CFeatures* f) 
		{ 
			invalidate_train();
			delete train_features; 
			train_features=f; 
			return true;
		}

		inline bool set_test_features(CFeatures* f) 
		{ 
			invalidate_test();
			delete test_features; 
			test_features=f; 
			return true;
		}

		void add_train_features(CFeatures* f);
		void add_test_features(CFeatures* f);

		void invalidate_train() ;
		void invalidate_test() ;
		

		bool load(CHAR* param);
		bool save(CHAR* param);
		bool clean(CHAR* param);

		bool reshape(CHAR* param);

		bool convert(CHAR* param);

		CSparseRealFeatures* convert_simple_real_to_sparse_real(CRealFeatures* src, CHAR* param);
		CStringFeatures<CHAR>* convert_simple_char_to_string_char(CCharFeatures* src, CHAR* param);
		CWordFeatures* convert_simple_char_to_simple_word(CCharFeatures* src, CHAR* param);
		CShortFeatures* convert_simple_char_to_simple_short(CCharFeatures* src, CHAR* param);
		CRealFeatures* convert_simple_char_to_simple_align(CCharFeatures* src,CHAR* param);
		CRealFeatures* convert_simple_word_to_simple_salzberg(CWordFeatures* src, CHAR* param);

		CStringFeatures<WORD>* convert_string_char_to_string_word(CStringFeatures<CHAR>* src, CHAR* param);
		CStringFeatures<ULONG>* convert_string_char_to_string_ulong(CStringFeatures<CHAR>* src, CHAR* param);
		CTOPFeatures* convert_string_word_to_simple_top(CStringFeatures<WORD>* src, CHAR* param);
		CFKFeatures* convert_string_word_to_simple_fk(CStringFeatures<WORD>* src, CHAR* param);

		CRealFeatures* convert_sparse_real_to_simple_real(CSparseRealFeatures* src, CHAR* param);

#ifdef HAVE_MINDY
		CMindyGramFeatures* CGUIFeatures::convert_string_char_to_mindy_grams(CStringFeatures<CHAR> *src, CHAR* param);
#endif

		bool set_ref_features(CHAR* param) ;

	protected:
		CGUI* gui;
		CFeatures *train_features;
		CFeatures *test_features;
		CFeatures *ref_features;
};
#endif
