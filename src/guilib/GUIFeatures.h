#ifndef __GUIFEATURES__H
#define __GUIFEATURES__H

#include "features/Labels.h"
#include "features/Features.h"
#include "hmm/Observation.h"

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

		bool set_features(char* param);

		/// apply the current preprocessor to train/test data
		/// (only useful when a feature matrix is available)
		bool preprocess(char* param);

		CFeatures *get_train_features() { return train_features; }
		CFeatures *get_test_features() { return test_features; }

		bool convert_full_to_sparse(char* param);
		bool convert_sparse_to_full(char* param);
		
		bool load(char* param);
		bool save(char* param);

		bool reshape(char* param);

	protected:
		bool preprocess_features(CFeatures* trainfeat, CFeatures* testfeat, bool force);
		bool preproc_all_features(CFeatures* f, bool force);

	protected:
		CGUI* gui;
		CFeatures *train_features;
		CFeatures *test_features;
		CObservation *train_obs;
		CObservation *test_obs;
};
#endif
