/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GUIFEATURES__H
#define __GUIFEATURES__H

#include "lib/config.h"

#ifndef HAVE_SWIG
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
#include "features/SparseFeatures.h"
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

		CSparseFeatures<DREAL>* convert_simple_real_to_sparse_real(CRealFeatures* src, CHAR* param);
		CStringFeatures<CHAR>* convert_simple_char_to_string_char(CCharFeatures* src, CHAR* param);
		CWordFeatures* convert_simple_char_to_simple_word(CCharFeatures* src, CHAR* param);
		CShortFeatures* convert_simple_char_to_simple_short(CCharFeatures* src, CHAR* param);
		CRealFeatures* convert_simple_char_to_simple_align(CCharFeatures* src,CHAR* param);
		CRealFeatures* convert_simple_word_to_simple_salzberg(CWordFeatures* src, CHAR* param);

		CStringFeatures<WORD>* convert_string_char_to_string_word(CStringFeatures<CHAR>* src, CHAR* param);
		CStringFeatures<ULONG>* convert_string_char_to_string_ulong(CStringFeatures<CHAR>* src, CHAR* param);
		CStringFeatures<WORD>* convert_string_byte_to_string_word(CStringFeatures<BYTE>* src, CHAR* param);
		CStringFeatures<ULONG>* convert_string_byte_to_string_ulong(CStringFeatures<BYTE>* src, CHAR* param);
		CTOPFeatures* convert_string_word_to_simple_top(CStringFeatures<WORD>* src, CHAR* param);
		CFKFeatures* convert_string_word_to_simple_fk(CStringFeatures<WORD>* src, CHAR* param);

		CRealFeatures* convert_sparse_real_to_simple_real(CSparseFeatures<DREAL>* src, CHAR* param);


		template <class CT, class ST> 
		CStringFeatures<ST>* convert_string_char_to_string_generic(CStringFeatures<CT>* src, CHAR* param)
		{
			CHAR target[1024]="";
			CHAR from_class[1024]="";
			CHAR from_type[1024]="";
			CHAR to_class[1024]="";
			CHAR to_type[1024]="";
			INT order=1;
			INT start=0;
			INT gap = 0 ;

			param=CIO::skip_spaces(param);

			if ((sscanf(param, "%s %s %s %s %s %d %d %d", target, from_class, from_type, to_class, to_type, &order, &start, &gap))<6)
			{
				CIO::message(M_ERROR, "see help for params (target, from_class, from_type, to_class, to_type, order, start, gap)\n");
				return NULL;
			}

			if ( (src) && ( (src->get_feature_class()) == C_STRING) )
			{
				//create dense features with 0 cache
				CIO::message(M_INFO, "converting CT STRING features to ST STRING ones (order=%i)\n",order);

				CStringFeatures<ST>* sf=new CStringFeatures<ST>(new CAlphabet(src->get_alphabet()));
				if (sf && sf->obtain_from_char_features(src, start, order, gap))
				{
					CIO::message(M_INFO, "conversion successful\n");
					return sf;
				}

				delete sf;
			}
			else
				CIO::message(M_ERROR, "no features of class/type STRING/CT available\n");

			return NULL;
		}

#ifdef HAVE_MINDY

		template <class CT>
		CMindyGramFeatures* convert_string_char_to_mindy_grams(CStringFeatures<CT> *src, CHAR* param)
		{
			CHAR mode[256]="", alph[256]="", embed[256]="",delim[256]="";
			INT nlen=-1;

			if (!src || !param) {
				CIO::message(M_ERROR, "invalid arguments: \"%s\"\n",param);
				return NULL;
			}

			if (sscanf(param, "%*s %*s %*s %*s %*s %255s %255s %255s", 
						mode, alph, embed) < 3) {
				CIO::message(M_ERROR, "too few arguments\n");
				return NULL;
			}

			if (!strcasecmp(mode, "words")) {
				if (sscanf(param, "%*s %*s %*s %*s %*s %*s %*s %*s %255s", 
							delim) < 1) {
					CIO::message(M_ERROR, "too few arguments\n");
					return NULL;
				}

				CIO::message(M_INFO, "Converting strings to Mindy words "
						"(a: %s, e: %s, d: '%s')\n", alph, embed, delim);                

				return new CMindyGramFeatures(src, alph, embed, delim);
			} else {
				if (sscanf(param, "%*s %*s %*s %*s %*s %*s %*s %*s %d", 
							&nlen) < 1) {
					CIO::message(M_ERROR, "too few arguments\n");
					return NULL;
				}

				CIO::message(M_INFO, "Converting strings to Mindy n-grams "
						"(a: %s, e: %s, n: %d)\n", alph, embed, nlen);                

				return new CMindyGramFeatures(src, alph, embed, nlen);
			} 
			
			return NULL;
		}
#endif

		bool set_ref_features(CHAR* param) ;

	protected:
		CGUI* gui;
		CFeatures *train_features;
		CFeatures *test_features;
		CFeatures *ref_features;
};
#endif
#endif
