/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __SIGNAL__H_
#define __SIGNAL__H_

#include "lib/config.h"
#include "base/SGObject.h"

#ifndef WIN32
#include <signal.h>
#define NUMTRAPPEDSIGS 2

#include "lib/python.h"

class CSignal : public CSGObject
{
public:
	CSignal();
	~CSignal();

	static void handler(int);

	static bool set_handler();
	static bool unset_handler();
	static void clear();
	static inline bool cancel_computations()
	{
#ifdef HAVE_PYTHON
	  if (PyErr_CheckSignals())
	  {
		  SG_SPRINT("\nImmediately return to matlab prompt / Prematurely finish computations / Do nothing (I/P/D)? ");
		  char answer=fgetc(stdin);

		  if (answer == 'I')
			  SG_SERROR("shogun stopped by SIGINT\n");
		  else if (answer == 'P')
		  {
			  PyErr_Clear();
			  cancel_computation=true;
		  }
		  else
			  SG_SPRINT("\n");
	  }
#endif
		return cancel_computation;
	}
protected:
	static int signals[NUMTRAPPEDSIGS];
	static struct sigaction oldsigaction[NUMTRAPPEDSIGS];
	static bool active;
	static bool cancel_computation;
};
#endif // WIN32
#endif // __SIGNAL__H_
