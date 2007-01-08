/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GUIMATH__H__ 
#define __GUIMATH__H__ 

#include "lib/config.h"

#ifndef HAVE_SWIG
class CGUI;

class CGUIMath
{
public:
	CGUIMath(CGUI *);
	void evaluate_results(DREAL* output, INT* label, INT total, FILE* outputfile=NULL, FILE* rocfile=NULL);
	void current_results(DREAL* output, INT* label, INT total, FILE* outputfile=NULL);

	void set_threshold(CHAR* input);
protected:
	CGUI* gui;
	DREAL threshold;
};
#endif //HAVE_SWIG
#endif
