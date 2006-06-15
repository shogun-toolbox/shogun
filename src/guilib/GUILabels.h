/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

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

		bool set_train_labels(CLabels* lab) { delete train_labels; train_labels=lab; return true;}
		bool set_test_labels(CLabels* lab) { delete test_labels; test_labels=lab; return true;}

		bool load(CHAR* param);
		bool save(CHAR* param);

	protected:
		CGUI* gui;
		CLabels *train_labels;
		CLabels *test_labels;
};
#endif
