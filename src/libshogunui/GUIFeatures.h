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

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/features/Labels.h>
#include <shogun/features/Features.h>
#include <shogun/features/RealFileFeatures.h>
#include <shogun/features/TOPFeatures.h>
#include <shogun/features/FKFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/features/CombinedDotFeatures.h>
#include <shogun/features/WDFeatures.h>
#include <shogun/features/ExplicitSpecFeatures.h>
#include <shogun/features/ImplicitWeightedSpecFeatures.h>

namespace shogun
{
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
			//invalidate_train();
			SG_UNREF(train_features);
			SG_REF(f);
			train_features=f;
			return true;
		}

		inline bool set_test_features(CFeatures* f)
		{ 
			//invalidate_test();
			SG_UNREF(test_features);
			SG_REF(f);
			test_features=f;
			return true;
		}

		void add_train_features(CFeatures* f);
		void add_test_features(CFeatures* f);
		void add_train_dotfeatures(CDotFeatures* f);
		void add_test_dotfeatures(CDotFeatures* f);

		/** delete last feature obj from combined features */
		bool del_last_feature_obj(char* target);

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
		/** reshape target feature matrix */
		bool reshape(char* target, int32_t num_feat, int32_t num_vec);

		/** get features for target to convert */
		CFeatures* get_convert_features(char* target);
		/** set convert(ed) features for target */
		bool set_convert_features(CFeatures* features, char* target);

		/* convert features from one class/type to another */
		CSparseFeatures<float64_t>* convert_simple_real_to_sparse_real(
			CSimpleFeatures<float64_t>* src);
		CStringFeatures<char>* convert_simple_char_to_string_char(
			CSimpleFeatures<char>* src);
		CSimpleFeatures<float64_t>* convert_simple_char_to_simple_align(
			CSimpleFeatures<char>* src,
			float64_t gap_cost=0);
		CSimpleFeatures<float64_t>* convert_simple_word_to_simple_salzberg(
			CSimpleFeatures<uint16_t>* src);

		CTOPFeatures* convert_string_word_to_simple_top(
			CStringFeatures<uint16_t>* src);
		CFKFeatures* convert_string_word_to_simple_fk(
			CStringFeatures<uint16_t>* src);
		CSimpleFeatures<float64_t>* convert_sparse_real_to_simple_real(
			CSparseFeatures<float64_t>* src);
		CExplicitSpecFeatures* convert_string_byte_to_spec_word(
				CStringFeatures<uint16_t>* src, bool use_norm);

		template <class CT, class ST>
		CStringFeatures<ST>* convert_string_char_to_string_generic(
			CStringFeatures<CT>* src,
			int32_t order=1, int32_t start=0, int32_t gap=0, char rev='f', CAlphabet* alpha=NULL)
		{
			if (src && src->get_feature_class()==C_STRING)
			{
				//create dense features with 0 cache
				SG_INFO("Converting CT STRING features to ST STRING ones (order=%i).\n",order);
				bool free_alpha=false;
				
				if (!alpha)
				{
					CAlphabet* a = src->get_alphabet();

					if ( a && a->get_alphabet() == DNA )
						alpha=new CAlphabet(RAWDNA);
					else
						alpha=new CAlphabet(a);

					free_alpha=true;
					SG_UNREF(a);
				}

				CStringFeatures<ST>* sf=new CStringFeatures<ST>(alpha);
				if (sf && sf->obtain_from_char_features(src, start, order, gap, rev=='r'))
				{
					SG_INFO("Conversion was successful.\n");
					return sf;
				}

				if (free_alpha)
					SG_UNREF(alpha);
				SG_UNREF(sf);
			}
			else
				SG_ERROR("No features of class/type STRING/CT available.\n");

			return NULL;
		}


		/** set reference features from target */
		bool set_reference_features(char* target);

		/** @return object name */
		inline virtual const char* get_name() const { return "GUIFeatures"; }

	protected:
		CSGInterface* ui;
		CFeatures *train_features;
		CFeatures *test_features;
		CFeatures *ref_features;
};
}
#endif
