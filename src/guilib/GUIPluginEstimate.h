#ifndef _GUIPLUGINESTIMATE_H__
#define _GUIPLUGINESTIMATE_H__ 

#include "classifier/PluginEstimate.h"
#include "features/Labels.h"

class CGUI;

class CGUIPluginEstimate
{

public:
	CGUIPluginEstimate(CGUI* g);
	~CGUIPluginEstimate();

	bool new_estimator(CHAR* param);
	bool train(CHAR* param);
	bool marginalized_train(CHAR* param);
	bool test(CHAR* param);
	bool load(CHAR* param);
	bool save(CHAR* param);

	inline CPluginEstimate* get_estimator() { return estimator; }

	CLabels* classify(CLabels* output=NULL);
	REAL classify_example(INT idx);

 protected:
	CGUI* gui;

	CPluginEstimate* estimator;
	REAL pos_pseudo;
	REAL neg_pseudo;
};
#endif
