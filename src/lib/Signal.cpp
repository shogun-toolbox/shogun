/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifndef WIN32
#include "lib/python.h"

#include <stdlib.h>
#include <signal.h>
#include <string.h>

#include "lib/io.h"
#include "lib/Signal.h"
#include "lib/matlab.h"
#include "lib/octave.h"
#include "lib/python.h"
#include "lib/r.h"

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
		SG_SERROR("error uninitalizing signal handler\n");
}

void CSignal::handler(int signal)
{
#if !defined(HAVE_SWIG) && (defined(HAVE_MATLAB) || defined(HAVE_OCTAVE) || defined(HAVE_R))
	if (signal == SIGINT)
	{
		SG_SPRINT("\nImmediately return to matlab prompt / Prematurely finish computations / Do nothing (I/P/D)? ");
		char answer=fgetc(stdin);

		if (answer == 'I')
		{
			unset_handler();
#if defined(HAVE_MATLAB)
			mexErrMsgTxt("sg stopped by SIGINT\n");
#elif defined(HAVE_OCTAVE)
			SG_PRINT("sg stopped by SIGINT\n");
			BEGIN_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
			octave_throw_interrupt_exception();
			END_INTERRUPT_IMMEDIATELY_IN_FOREIGN_CODE;
#elif defined(HAVE_PYTHON)
			SG_PRINT("sg stopped by SIGINT\n");
			PyErr_SetInterrupt();
			PyErr_CheckSignals();
#elif defined(HAVE_R)
			R_Suicide((char*) "sg stopped by SIGINT\n");
#endif

		}
		else if (answer == 'P')
			cancel_computation=true;
		else
			SG_SPRINT("\n");
	}
	else if (signal == SIGURG)
		cancel_computation=true;
	else
		SG_SERROR("unknown signal %d received\n", signal);
#else
	SG_SPRINT("\n");
	SG_SERROR("sg stopped by SIGINT\n");
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

#ifndef __INTERIX
		act.sa_sigaction=NULL; //just in case
#endif
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
				SG_SERROR("error uninitalizing signal handler for signal %d\n", signals[i]);
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
#endif //WIN32
