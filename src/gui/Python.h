/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifdef HAVE_PYTHON

#ifndef __GFPYTHON__
#define __GFPYTHON__

extern "C" {
#include <Python.h>

	PyMODINIT_FUNC initsg(void);
	void exitsg(void);
}
#endif
#endif
