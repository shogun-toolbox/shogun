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

#ifndef __GUIFEATURES__H
#define __GUIFEATURES__H

#include "lib/config.h"
#include "base/SGObject.h"

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

class CGUIFeatures : public CSGObject
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

		void invalidate_train();
		void invalidate_test();

		/** load features from file */
		bool load(CHAR* filename, CHAR* fclass, CHAR* type,
			CHAR* target, INT size, INT comp_features);
		/** save features to file */
		bool save(CHAR* filename, CHAR* type, CHAR* target);
		/** clean/r features */
		bool clean(CHAR* target);
		/** obtain feature buy sliding window */
		bool obtain_by_sliding_window(CHAR* target, INT winsize, INT shift, INT skip);
		/** reshape target feature matrix */
		bool reshape(CHAR* target, INT num_feat, INT num_vec);

		/** get features for target to convert */
		CFeatures* get_convert_features(CHAR* target);
		/** set convert(ed) features for target */
		bool set_convert_features(CFeatures* features, CHAR* target);

		/* convert features from one class/type to another */
		CSparseFeatures<DREAL>* convert_simple_real_to_sparse_real(CRealFeatures* src);
		CStringFeatures<CHAR>* convert_simple_char_to_string_char(CCharFeatures* src);
		CWordFeatures* convert_simple_char_to_simple_word(
			CCharFeatures* src, INT order=1, INT start=0, INT gap=0);
		CShortFeatures* convert_simple_char_to_simple_short(CCharFeatures* src, INT order=1, INT start=0, INT gap=0);
		CRealFeatures* convert_simple_char_to_simple_align(CCharFeatures* src, DREAL gap_cost=0);
		CRealFeatures* convert_simple_word_to_simple_salzberg(CWordFeatures* src);

		CStringFeatures<WORD>* convert_string_char_to_string_word(CStringFeatures<CHAR>* src);
		CStringFeatures<ULONG>* convert_string_char_to_string_ulong(CStringFeatures<CHAR>* src);
		CTOPFeatures* convert_string_word_to_simple_top(CStringFeatures<WORD>* src);
		CFKFeatures* convert_string_word_to_simple_fk(CStringFeatures<WORD>* src);

		CRealFeatures* convert_sparse_real_to_simple_real(CSparseFeatures<DREAL>* src);

		template <class CT, class ST>
		CStringFeatures<ST>* convert_string_char_to_string_generic(CStringFeatures<CT>* src, INT order=1, INT start=0, INT gap=0, CHAR rev='f')
		{
			if (src && src->get_feature_class()==C_STRING)
			{
				//create dense features with 0 cache
				SG_INFO("Converting CT STRING features to ST STRING ones (order=%i).\n",order);

				CStringFeatures<ST>* sf=new CStringFeatures<ST>(new CAlphabet(src->get_alphabet()));
				if (sf && sf->obtain_from_char_features(src, start, order, gap, rev=='r'))
				{
					SG_INFO("Conversion was successful.\n");
					return sf;
				}

				delete sf;
			}
			else
				SG_ERROR("No features of class/type STRING/CT available.\n");

			return NULL;
		}

#ifdef HAVE_MINDY
		template <class CT>
		CMindyGramFeatures* convert_string_char_to_mindy_grams(
			CStringFeatures<CT> *src, CHAR* alph, CHAR* embed,
			INT nlen, CHAR* delim, DREAL maxv)
		{
			if (!src || !aplh || !embed || !delim) {
				SG_ERROR("Invalid arguments.\n");
				return NULL;
			}

			SG_INFO("Converting string to Mindy features "
					"(a: %s, e: %s, n: %d, d: '%s', m: %f)\n",
					alph, embed, nlen, delim, maxv);

			CMindyGramFeatures* mgf=new CMindyGramFeatures(
				alph, embed, delim, nlen);
			mgf->import_features(src);
			mgf->trim_max(maxv);
			return mgf;
		}
#endif

		/** set reference features from target */
		bool set_reference_features(CHAR* target);

	protected:
		CGUI* gui;
		CFeatures *train_features;
		CFeatures *test_features;
		CFeatures *ref_features;
};
#endif
#endif
