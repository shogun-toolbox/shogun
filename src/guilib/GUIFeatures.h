#ifndef __GUIFEATURES__H
#define __GUIFEATURES__H

#include "features/Labels.h"
#include "features/Features.h"

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

		/// apply the current preprocessor to train/test data
		/// (only useful when a feature matrix is available)
		bool preprocess(CHAR* param);

		inline CFeatures *get_train_features() { return train_features; }
		inline CFeatures *get_test_features() { return test_features; }

		inline void set_train_features(CFeatures* f) { delete train_features; train_features=f; }
		inline void set_test_features(CFeatures* f) { delete test_features; test_features=f; }

		void add_train_features(CFeatures* f);
		void add_test_features(CFeatures* f);

		
		bool load(CHAR* param);
		bool save(CHAR* param);

		bool reshape(CHAR* param);

		bool convert(CHAR* param);

		/// obsolete use the more generic convert function
		bool convert_full_to_sparse(CHAR* param);
		bool convert_sparse_to_full(CHAR* param);
		bool convert_char_to_word(CHAR* param);
		bool convert_char_to_short(CHAR* param);

		bool alignment_char(CHAR* param) ;
		bool set_ref_features(CHAR* param) ;
	protected:
		bool preprocess_features(CFeatures* trainfeat, CFeatures* testfeat, bool force);
		bool preproc_all_features(CFeatures* f, bool force);

	protected:
		CGUI* gui;
		CFeatures *train_features;
		CFeatures *test_features;
		CFeatures *ref_features;
};
#endif
