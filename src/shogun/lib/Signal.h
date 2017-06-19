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
#include <shogun/base/SGObject.h>

namespace shogun
{
	/** @brief Class Signal implements signal handling to e.g. allow CTRL+C to
	 * cancel a long running process.
	 *
	 * -# A signal handler is attached to trap the SIGINT signal.
	 *  Pressing CTRL+C or sending the SIGINT (kill ...) signal to the shogun
	 *  process will make shogun print a message asking the user to choose an
	 *  option bewteen: immediately exit the running method and fall back to
	 *  the command line, prematurely stop the current algoritmh and do nothing.
	 */
	class CSignal : public CSGObject
	{
	public:
		CSignal();
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

		/** Cancel computations
		 *
		 * @return if computations should be cancelled
		 */
		static inline bool cancel_computations()
		{
			return false;
		}

		/** Enable signal handler
		*/
		void enable_handler()
		{
			m_active = true;
		}

		/** @return object name */
		virtual const char* get_name() const { return "Signal"; }

	private:
		/** Active signal */
		static bool m_active;

		/** SIGINT Observable */
		static rxcpp::connectable_observable<int> m_sigint_observable;

		/** SIGURG Observable */
		static rxcpp::connectable_observable<int> m_sigurg_observable;
};
}
#endif // __SIGNAL__H_
