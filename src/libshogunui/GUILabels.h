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

#ifndef __GUILABELS__H_
#define __GUILABELS__H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/features/Labels.h>

namespace shogun
{
class CSGInterface;

class CGUILabels : public CSGObject
{
	public:
		CGUILabels(CSGInterface* interface);
		~CGUILabels();

		CLabels *get_train_labels() { return train_labels; }
		CLabels *get_test_labels() { return test_labels; }

		bool set_train_labels(CLabels* lab) { SG_UNREF(train_labels); SG_REF(lab); train_labels=lab; return true;}
		bool set_test_labels(CLabels* lab) { SG_UNREF(test_labels); SG_REF(lab); test_labels=lab; return true;}

		/** load labels from file */
		bool load(char* filename, char* target);
		bool save(char* param);

		/** @return object name */
		inline virtual const char* get_name() const { return "GUILabels"; }

	protected:
		CSGInterface* ui;
		CLabels *train_labels;
		CLabels *test_labels;
};
}
#endif
