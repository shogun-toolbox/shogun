/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2017 Giovanni De Toni
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __SIGNAL__H_
#define __SIGNAL__H_

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
	class CSignal : CSGObject
	{
	public:
		typedef rxcpp::subjects::subject<int> SGSubjectS;
		typedef rxcpp::observable<int, rxcpp::dynamic_observable<int>>
		    SGObservableS;
		typedef rxcpp::subscriber<int,
		                          rxcpp::observer<int, void, void, void, void>>
		    SGSubscriberS;

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
		SGObservableS* get_observable()
		{
			return m_observable;
		};
#endif

#ifndef SWIG // SWIG should skip this part
		     /**
		     * Get subscriber
		     * @return RxCpp subscriber
		     */
		SGSubscriberS* get_subscriber()
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
		static void reset_handler();

		/** @return object name */
		virtual const char* get_name() const { return "Signal"; }

	private:
		/** Active signal */
		static bool m_active;

	public:
		/** Observable */
		static SGSubjectS* m_subject;
		static SGObservableS* m_observable;
		static SGSubscriberS* m_subscriber;
};
}
#endif // __SIGNAL__H_
