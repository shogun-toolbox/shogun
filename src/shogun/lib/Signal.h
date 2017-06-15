/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __SIGNAL__H_
#define __SIGNAL__H_

#include <rxcpp/rx-includes.hpp>
#include <shogun/lib/config.h>

#if defined(__MINGW64__) || defined(_MSC_VER)
typedef unsigned long sigset_t;
#endif
#if defined(__MINGW32__) && !defined(__MINGW64__)
typedef int sigset_t;
#endif

#ifndef SIGURG
#define SIGURG  -16
#endif

#if defined(__MINGW64__) || defined(_MSC_VER) || defined(__MINGW32__)
typedef void Sigfunc (int);

struct sigaction {
	Sigfunc *sa_handler;
	sigset_t sa_mask;
	int sa_flags;
};

#define sigemptyset(ptr) (*(ptr) = 0)
#define sigfillset(ptr) ( *(ptr) = ~(sigset_t)0,0)

int sigaddset(sigset_t*, int);
int sigaction(int signo, const struct sigaction *act, struct sigaction *oact);
#endif

#ifndef DISABLE_CANCEL_CALLBACK
namespace shogun
{
extern void (*sg_cancel_computations)(bool &delayed, bool &immediately);
}
#endif

#include <shogun/lib/ShogunException.h>
#include <shogun/base/SGObject.h>

#include <csignal>
#define NUMTRAPPEDSIGS 2

namespace shogun
{
/** @brief Class Signal implements signal handling to e.g. allow ctrl+c to cancel a
 * long running process.
 *
 * This is done in two ways:
 *
 * -# A signal handler is attached to trap the SIGINT and SIGURG signal.
 *  Pressing ctrl+c or sending the SIGINT (kill ...) signal to the shogun
 *  process will make shogun print a message asking to immediately exit the
 *  running method and to fall back to the command line.
 * -# When an URG signal is received or ctrl+c P is pressed shogun will
 *  prematurely stop a method and continue execution. For example when an SVM
 *  solver takes a long time without progressing much, one might still be
 *  interested in the result and should thus send SIGURG or interactively
 *  prematurely stop the method
 */
class CSignal : public CSGObject
{
	public:
		CSignal();
		CSignal(bool active);
		virtual ~CSignal();

		/** Signal handler. Need to be registered with std::signal.
		 *
		 * @param signal signal number
		 */
		static void handler(int signal);

		/**
		 * Get SIGINT observable
		 * @return observable
		 */
		rxcpp::connectable_observable<int> get_SIGINT_observable();

		/**
		* Get SIGURG observable
		* @ return observable
		*/
		rxcpp::connectable_observable<int> get_SIGURG_observable();

		/** cancel computations
		 *
		 * @return if computations should be cancelled
		 */
		static inline bool cancel_computations()
		{
			return false;
		}

		/** @return object name */
		virtual const char* get_name() const { return "Signal"; }

	private:
		/** signals; handling external lib  */
		static int signals[NUMTRAPPEDSIGS];

		/** signal actions */
		static struct sigaction oldsigaction[NUMTRAPPEDSIGS];

		/** active signal */
		bool m_active;

		static rxcpp::connectable_observable<int> m_sigint_observable;
		static rxcpp::connectable_observable<int> m_sigurg_observable;
};
}
#endif // __SIGNAL__H_
