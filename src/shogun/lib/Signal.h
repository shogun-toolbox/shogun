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
	/**
	 * Possible Shogun signal types.
	 */
	enum sg_signals_types
	{
		SG_BLOCK_COMP,
		SG_PAUSE_COMP
	};

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

#ifndef SWIG // SWIG should skip this part
		     /**
		     * Get observable
		     * @return RxCpp observable
		     */
		rxcpp::observable<int> get_observable()
		{
			return m_observable;
		};
#endif

#ifndef SWIG // SWIG should skip this part
		     /**
		     * Get subscriber
		     * @return RxCpp subscriber
		     */
		rxcpp::subscriber<int> get_subscriber()
		{
			return m_subscriber;
		};
#endif

		/** Enable signal handler
		*/
		void enable_handler()
		{
			m_active = true;
		}
		/**
		 * Reset handler in case of multiple instantiation
		 */
		static void reset_handler()
		{
			m_subject = rxcpp::subjects::subject<int>();
			m_observable = m_subject.get_observable();
			m_subscriber = m_subject.get_subscriber();
		}

		/** @return object name */
		virtual const char* get_name() const { return "Signal"; }

	private:
		/** Active signal */
		static bool m_active;

		/** Observable */
		static rxcpp::subjects::subject<int> m_subject;
		static rxcpp::observable<int> m_observable;
		static rxcpp::subscriber<int> m_subscriber;
};
}
#endif // __SIGNAL__H_
