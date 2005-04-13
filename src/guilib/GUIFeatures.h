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

		inline void set_train_features(CFeatures* f) 
			{ 
				invalidate_train() ;
				delete train_features; 
				train_features=f; 
			} ;
		
		inline void set_test_features(CFeatures* f) 
			{ 
				invalidate_test() ;
				delete test_features; 
				test_features=f; 
			} ;

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

		bool set_ref_features(CHAR* param) ;

	protected:
		CGUI* gui;
		CFeatures *train_features;
		CFeatures *test_features;
		CFeatures *ref_features;
};
#endif
