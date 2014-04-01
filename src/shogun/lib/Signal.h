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

#include <shogun/lib/config.h>

#ifndef DISABLE_CANCEL_CALLBACK
namespace shogun
{
extern void (*sg_cancel_computations)(bool &delayed, bool &immediately);
}
#endif

#include <shogun/lib/config.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/base/SGObject.h>

#ifndef WIN32
#include <signal.h>
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
		/** default constructor */
		CSignal();
		virtual ~CSignal();

		/** handler
		 *
		 * @param signal signal number
		 */
		static void handler(int signal);

		/** set handler
		 *
		 * @return if setting was successful
		 */
		static bool set_handler();

		/** unset handler
		 *
		 * @return if unsetting was successful
		 */
		static bool unset_handler();

		/** clear signals */
		static void clear();

		/** clear cancel flag signals */
		static void clear_cancel();

		/** set cancel flag signals */
		static void set_cancel(bool immediately=false);

		/** cancel computations
		 *
		 * @return if computations should be cancelled
		 */
		static inline bool cancel_computations()
		{
#ifndef DISABLE_CANCEL_CALLBACK
			if (sg_cancel_computations)
				sg_cancel_computations(cancel_computation, cancel_immediately);
#endif
			if (cancel_immediately)
				throw ShogunException("Computations have been cancelled immediately");

			return cancel_computation;
		}

		/** @return object name */
		virtual const char* get_name() const { return "Signal"; }

	protected:
		/** signals; handling external lib  */
		static int signals[NUMTRAPPEDSIGS];

		/** signal actions */
		static struct sigaction oldsigaction[NUMTRAPPEDSIGS];

		/** active signal */
		static bool active;

		/** if computation should be cancelled */
		static bool cancel_computation;

		/** if shogun should return ASAP */
		static bool cancel_immediately;
};
}
#endif // WIN32
#endif // __SIGNAL__H_
