/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <csignal>
#include <stdlib.h>

#include <rxcpp/rx.hpp>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>

using namespace shogun;
using namespace rxcpp;

bool CSignal::m_active = false;

rxcpp::connectable_observable<int> CSignal::m_sigint_observable =
    rxcpp::observable<>::create<int>([](rxcpp::subscriber<int> s) {
	    s.on_completed();
	}).publish();

rxcpp::connectable_observable<int> CSignal::m_sigurg_observable =
    rxcpp::observable<>::create<int>([](rxcpp::subscriber<int> s) {
	    s.on_next(1);
	}).publish();

CSignal::CSignal()
{
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
	/* If the handler is not enabled, then return */
	if (!m_active)
		return;

	if (signal == SIGINT)
	{
		SG_SPRINT(
		    "\n[ShogunSignalHandler] Immediately return to prompt / "
		    "Prematurely finish "
		    "computations / Do nothing (I/P/D)? ")
		char answer = fgetc(stdin);
		switch (answer)
		{
		case 'I':
			SG_SPRINT("[ShogunSignalHandler] Killing the application...\n");
			m_sigint_observable.connect();
			exit(0);
			break;
		case 'P':
			SG_SPRINT(
			    "[ShogunSignalHandler] Terminating"
			    " prematurely current algorithm...\n");
			m_sigurg_observable.connect();
			break;
		default:
			SG_SPRINT("[ShogunSignalHandler] Continuing...\n")
			break;
		}
	}
	else
	{
		SG_SPRINT("[ShogunSignalHandler] Unknown signal %d received\n", signal)
	}
}
