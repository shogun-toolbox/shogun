#ifndef __GUIFEATURES__H
#define __GUIFEATURES__H

#include "features/Features.h"
#include "features/TOPFeatures.h"

class CGUI;

class CGUIFeatures
{
	public:
		enum E_FEATURE_TYPE
		{
			TOP,
			FK
		};

		CGUIFeatures(CGUI *);
		~CGUIFeatures();

		bool set_features(char* param);

		CFeatures *get_train_features() { return train_features; }
		CFeatures *get_test_features() { return test_features; }

	protected:
		CGUI* gui;
		CFeatures *train_features;
		CFeatures *test_features;
		E_FEATURE_TYPE type;
};
#endif
