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
#include <shogun/labels/Labels.h>

namespace shogun
{
class CSGInterface;

/** @brief UI labels */
class CGUILabels : public CSGObject
{
	public:
		/** constructor */
		CGUILabels() {};
		/** constructor
		 * @param interface
		 */
		CGUILabels(CSGInterface* interface);
		/** destructor */
		~CGUILabels();

		/** get train labels */
		CLabels *get_train_labels() { return train_labels; }
		/** get test labels */
		CLabels *get_test_labels() { return test_labels; }

		/** set train labels
		 * @param lab
		 */
		bool set_train_labels(CLabels* lab) { SG_REF(lab); SG_UNREF(train_labels); train_labels=lab; return true;}
		/** set test labels
		 * @param lab
		 */
		bool set_test_labels(CLabels* lab) { SG_REF(lab); SG_UNREF(test_labels); test_labels=lab; return true;}

		/** load labels from file
		 * @param filename
		 * @param target
		 */
		bool load(char* filename, char* target);
		/** save
		 * @param param
		 */
		bool save(char* param);

		/** infer labels from array
		 *
		 * @param lab array
		 * @param len length of array
		 * @return labels
		 */
		CLabels* infer_labels(float64_t* lab, int32_t len);

		/** @return object name */
		virtual const char* get_name() const { return "GUILabels"; }

	protected:
		/** ui */
		CSGInterface* ui;
		/** train labels */
		CLabels *train_labels;
		/** test labels */
		CLabels *test_labels;
};
}
#endif
