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

class CSGInterface;

class CGUIFeatures : public CSGObject
{
	enum EFeatureType
	{
		Simple,
		Sparse
	};

	public:
		CGUIFeatures(CSGInterface* interface);
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

		/** delete last features from combined features */
		bool del_last_features(char* target);

		void invalidate_train();
		void invalidate_test();

		/** load features from file */
		bool load(
			char* filename, char* fclass, char* type, char* target,
			int32_t size, int32_t comp_features);
		/** save features to file */
		bool save(char* filename, char* type, char* target);
		/** clean/r features */
		bool clean(char* target);
		/** obtain feature by sliding window */
		bool obtain_by_sliding_window(
			char* target, int32_t winsize, int32_t shift, int32_t skip=0);
		/** reshape target feature matrix */
		bool reshape(char* target, int32_t num_feat, int32_t num_vec);

		/** get features for target to convert */
		CFeatures* get_convert_features(char* target);
		/** set convert(ed) features for target */
		bool set_convert_features(CFeatures* features, char* target);

		/* convert features from one class/type to another */
		CSparseFeatures<float64_t>* convert_simple_real_to_sparse_real(
			CRealFeatures* src);
		CStringFeatures<char>* convert_simple_char_to_string_char(
			CCharFeatures* src);
		CWordFeatures* convert_simple_char_to_simple_word(
			CCharFeatures* src,
			int32_t order=1, int32_t start=0, int32_t gap=0);
		CShortFeatures* convert_simple_char_to_simple_short(
			CCharFeatures* src,
			int32_t order=1, int32_t start=0, int32_t gap=0);
		CRealFeatures* convert_simple_char_to_simple_align(
			CCharFeatures* src,
			float64_t gap_cost=0);
		CRealFeatures* convert_simple_word_to_simple_salzberg(
			CWordFeatures* src);

		CTOPFeatures* convert_string_word_to_simple_top(
			CStringFeatures<uint16_t>* src);
		CFKFeatures* convert_string_word_to_simple_fk(
			CStringFeatures<uint16_t>* src);
		CRealFeatures* convert_sparse_real_to_simple_real(
			CSparseFeatures<float64_t>* src);

		template <class CT, class ST>
		CStringFeatures<ST>* convert_string_char_to_string_generic(
			CStringFeatures<CT>* src,
			int32_t order=1, int32_t start=0, int32_t gap=0, char rev='f')
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
			CStringFeatures<CT> *src, char* alph, char* embed,
			int32_t nlen, char* delim, float64_t maxv)
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
		bool set_reference_features(char* target);

	protected:
		CSGInterface* ui;
		CFeatures *train_features;
		CFeatures *test_features;
		CFeatures *ref_features;
};
#endif
#endif
