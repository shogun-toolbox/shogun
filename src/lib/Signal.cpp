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

#include "lib/config.h"

#ifdef HAVE_PYTHON
#include <Python.h>
#endif

#include <stdlib.h>
#include <signal.h>
#include <string.h>

#include "lib/io.h"
#include "lib/Signal.h"


int CSignal::signals[NUMTRAPPEDSIGS]={SIGINT, SIGURG};
struct sigaction CSignal::oldsigaction[NUMTRAPPEDSIGS];
bool CSignal::active=false;
bool CSignal::cancel_computation=false;

CSignal::CSignal()
{
}

CSignal::~CSignal()
{
	if (!unset_handler())
		CIO::message(M_ERROR, "error uninitalizing signal handler\n");
}

void CSignal::handler(int signal)
{
#ifdef HAVE_MATLAB
	if (signal == SIGINT)
	{
		CIO::message(M_MESSAGEONLY, "\nImmediately return to matlab prompt / Prematurely finish computations / Do nothing (I/P/D)? ");
		char answer=fgetc(stdin);

		if (answer == 'I')
		{
			unset_handler();
			CIO::message(M_ERROR, "sg stopped by SIGINT\n");
		}
		else if (answer == 'P')
			cancel_computation=true;
		else
			CIO::message(M_MESSAGEONLY, "\n");
	}
	else if (signal == SIGURG)
		cancel_computation=true;
	else
		CIO::message(M_ERROR, "unknown signal %d received\n", signal);
#else
	CIO::message(M_MESSAGEONLY, "\n");
	CIO::message(M_ERROR, "sg stopped by SIGINT\n");
	unset_handler();
	exit(0);
#endif
}

bool CSignal::set_handler()
{
	if (!active)
	{
		struct sigaction act;
		sigset_t st;

		sigemptyset(&st);

		act.sa_sigaction=NULL; //just in case
		act.sa_handler=CSignal::handler;
		act.sa_mask = st;
		act.sa_flags = 0;

		for (INT i=0; i<NUMTRAPPEDSIGS; i++)
		{
			if (sigaction(signals[i], &act, &oldsigaction[i]))
			{
				for (INT j=i-1; j>=0; j--)
					sigaction(signals[i], &oldsigaction[i], NULL);

				clear();
				return false;
			}
		}

		active=true;
		return true;
	}
	else
		return false;
}

bool CSignal::unset_handler()
{
	if (active)
	{
		bool result=true;

		for (INT i=0; i<NUMTRAPPEDSIGS; i++)
		{
			if (sigaction(signals[i], &oldsigaction[i], NULL))
			{
				CIO::message(M_ERROR, "error uninitalizing signal handler for signal %d\n", signals[i]);
				result=false;
			}
		}

		if (result)
			clear();

		return result;
	}
	else
		return false;
}

void CSignal::clear()
{
	cancel_computation=false;
	active=false;
	memset(&CSignal::oldsigaction, 0, sizeof(CSignal::oldsigaction));
}
