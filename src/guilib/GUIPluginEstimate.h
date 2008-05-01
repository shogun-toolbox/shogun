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

#ifndef _GUIPLUGINESTIMATE_H__
#define _GUIPLUGINESTIMATE_H__ 

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "base/SGObject.h"

#include "classifier/PluginEstimate.h"
#include "features/Labels.h"

class CGUI;

class CGUIPluginEstimate : public CSGObject
{

public:
	CGUIPluginEstimate(CGUI* g);
	~CGUIPluginEstimate();

	/** create new estimator */
	bool new_estimator(DREAL pos, DREAL neg);
	/** train estimator */
	bool train();
	bool marginalized_train(CHAR* param);
	/** test estimator */
	bool test(CHAR* filename_out, CHAR* filename_roc);
	bool load(CHAR* param);
	bool save(CHAR* param);

	inline CPluginEstimate* get_estimator() { return estimator; }

	CLabels* classify(CLabels* output=NULL);
	DREAL classify_example(INT idx);

 protected:
	CGUI* gui;

	CPluginEstimate* estimator;
	DREAL pos_pseudo;
	DREAL neg_pseudo;
};
#endif //HAVE_SWIG
#endif
