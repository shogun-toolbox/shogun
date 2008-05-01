/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GUIMATH__H__ 
#define __GUIMATH__H__ 

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "base/SGObject.h"

class CSGInterface;

class CGUIMath : public CSGObject
{
	public:
		CGUIMath(CSGInterface* interface);
		void evaluate_results(DREAL* output, INT* label, INT total, FILE* outputfile=NULL, FILE* rocfile=NULL);
		void current_results(DREAL* output, INT* label, INT total, FILE* outputfile=NULL);

		void set_threshold(DREAL value);

	protected:
		CSGInterface* ui;
		DREAL threshold;
};
#endif //HAVE_SWIG
#endif
