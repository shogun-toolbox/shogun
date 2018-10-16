/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Soeren Sonnenburg, Thoralf Klein, Viktor Gal
 */

#include <csignal>
#include <stdlib.h>

#include <rxcpp/rx-lite.hpp>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>

using namespace shogun;
using namespace rxcpp;

bool CSignal::m_active = true;
bool CSignal::m_interactive = true;
CSignal::SGSubjectS* CSignal::m_subject = new rxcpp::subjects::subject<int>();

CSignal::SGObservableS* CSignal::m_observable =
    new CSignal::SGObservableS(CSignal::m_subject->get_observable());
CSignal::SGSubscriberS* CSignal::m_subscriber =
    new CSignal::SGSubscriberS(CSignal::m_subject->get_subscriber());

CSignal::CSignal()
{
}

CSignal::~CSignal()
{
}

int CSignal::interactive_signal()
{
	int what_action = -1;
	SG_SPRINT(
	    "\n[ShogunSignalHandler] "
	    "Immediately return to prompt / "
	    "Prematurely finish computations / "
	    "Pause current computation / "
	    "Do nothing (I/C/P/D)? ")
	char answer = getchar();
	getchar();
	switch (answer)
	{
	case 'I':
		what_action = SIGINT;
		break;
	case 'C':
		what_action = SIGQUIT;
		break;
	case 'P':
		what_action = SIGTSTP;
		break;
	default:
		break;
	}
	return what_action;
}

void CSignal::handler(int signal)
{
	/* If the handler is not enabled exit */
	if (!m_active)
		exit(-1);

	/* If we are using interactive mode, ask the user what to do */
	if (m_interactive)
		signal = interactive_signal();

	/* Check which signal we have received */
	switch (signal)
	{
	case -1:
		SG_SPRINT("[ShogunSignalHandler] Continuing...\n");
	case SIGINT:
		SG_SPRINT("[ShogunSignalHandler] Killing the application...\n");
		m_subscriber->on_completed();
		exit(0);
		break;
	case SIGQUIT:
		SG_SPRINT(
		    "[ShogunSignalHandler] Terminating"
		    " prematurely current algorithm...\n");
		m_subscriber->on_next(SG_BLOCK_COMP);
		break;
	case SIGTSTP:
		SG_SPRINT("[ShogunSignalHandler] Pausing current computation...\n")
		m_subscriber->on_next(SG_PAUSE_COMP);
		break;
	default:
		SG_SPRINT("[ShogunSignalHandler] Unknown signal %d received\n", signal)
		break;
	}
}

void CSignal::reset_handler()
{
	delete m_subject;
	delete m_observable;
	delete m_subscriber;

	m_subject = new rxcpp::subjects::subject<int>();
	m_observable = new CSignal::SGObservableS(m_subject->get_observable());
	m_subscriber = new CSignal::SGSubscriberS(m_subject->get_subscriber());
}
