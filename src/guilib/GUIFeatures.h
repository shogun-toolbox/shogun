#ifndef __GUIFEATURES__H
#define __GUIFEATURES__H

#include "features/Labels.h"
#include "features/Features.h"
#include "hmm/Observation.h"

class CGUI;

class CGUIFeatures
{
	public:
		CGUIFeatures(CGUI *);
		~CGUIFeatures();

		bool set_features(char* param);

		/// apply the current preprocessor to train/test data
		/// (only useful when a feature matrix is available)
		bool preprocess(char* param);

		CFeatures *get_train_features() { return train_features; }
		CFeatures *get_test_features() { return test_features; }
		
		bool load(char* param);
		bool save(char* param);

		bool reshape(char* param);

	protected:
		bool preprocess_features(CFeatures* trainfeat, CFeatures* testfeat, bool force);

	protected:
		CGUI* gui;
		CFeatures *train_features;
		CFeatures *test_features;
		CObservation *train_obs;
		CObservation *test_obs;
};
#endif
