#ifndef __GUILABELS__H_
#define __GUILABELS__H_

#include "features/Labels.h"

class CGUI;

class CGUILabels
{
	public:
		CGUILabels(CGUI *);
		~CGUILabels();

		CLabels *get_train_labels() { return train_labels; }
		CLabels *get_test_labels() { return test_labels; }

		bool load(char* param);
		bool save(char* param);

	protected:
		CGUI* gui;
		CLabels *train_labels;
		CLabels *test_labels;
};
#endif
