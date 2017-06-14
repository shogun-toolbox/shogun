/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/config.h>

#include <stdlib.h>

#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/base/init.h>
#include <rxcpp/rx-includes.hpp>
#include <rxcpp/rx.hpp>

using namespace shogun;
using namespace rxcpp;

int CSignal::signals[NUMTRAPPEDSIGS]={SIGINT, SIGURG};
struct sigaction CSignal::oldsigaction[NUMTRAPPEDSIGS];

rxcpp::connectable_observable<int> CSignal::m_sigint_observable = rxcpp::observable<>::create<int>(
		[](rxcpp::subscriber<int> s){
			s.on_completed();
		}
).publish();

rxcpp::connectable_observable<int> CSignal::m_sigurg_observable = rxcpp::observable<>::create<int>(
	   [](rxcpp::subscriber<int> s){
		   s.on_next(1);
	   }
).publish();

CSignal::CSignal()
: CSGObject()
{
	// Set if the signal handler is active or not
	m_active = true;
}

CSignal::CSignal(bool active)
: CSGObject()
{
	// Set if the signal handler is active or not
	m_active = active;
}

CSignal::~CSignal()
{
}

rxcpp::connectable_observable<int> CSignal::get_SIGINT_observable()
{
	return m_sigint_observable;
}

rxcpp::connectable_observable<int> CSignal::get_SIGURG_observable()
{
	return m_sigurg_observable;
}

void CSignal::handler(int signal)
{
	if (signal == SIGINT)
	{
		//SG_SPRINT("\nImmediately return to prompt / Prematurely finish computations / Do nothing (I/P/D)? ")
		//char answer=fgetc(stdin);
		/*switch (answer){
			case 'I':
				m_sigint_observable.connect();
				break;
			case 'P':
				m_sigurg_observable.connect();
				break;
			default:
				SG_SPRINT("Continuing...\n")
				break;
		}*/
		SG_SPRINT("Killing the application...\n");
		m_sigint_observable.connect();

	}
	else if (signal == SIGURG)
		m_sigurg_observable.connect();
	else
		SG_SPRINT("unknown signal %d received\n", signal)
}

#if defined(__MINGW64__) || defined(_MSC_VER) || defined(__MINGW32__)
#define SIGBAD(signo) ( (signo) <=0 || (signo) >=NSIG)
Sigfunc *handlers[NSIG]={0};

int sigaddset(sigset_t *set, int signo)
{
	if (SIGBAD(signo)) {
		errno = EINVAL;
		return -1;
	}
	*set |= 1 << (signo-1);
	return 0;
}

int sigaction(int signo, const struct sigaction *act, struct sigaction *oact)
{
	if (SIGBAD(signo)) {
		errno = EINVAL;
		return -1;
	}

	if(oact){
			oact->sa_handler = handlers[signo];
			oact->sa_mask = 0;
			oact->sa_flags =0;
	}
	if (act)
		handlers[signo]=act->sa_handler;

	return 0;
}
#endif
