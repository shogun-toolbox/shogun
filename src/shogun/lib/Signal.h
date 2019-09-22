/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Soeren Sonnenburg, Viktor Gal, Yuyu Zhang, 
 *          Thoralf Klein, Evan Shelhamer, Sergey Lisitsyn
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
	class Signal
	{
	public:
		typedef rxcpp::subjects::subject<int> SGSubjectS;
		typedef rxcpp::observable<int, rxcpp::dynamic_observable<int>>
		    SGObservableS;
		typedef rxcpp::subscriber<int,
		                          rxcpp::observer<int, void, void, void, void>>
		    SGSubscriberS;

		Signal();
		virtual ~Signal();

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

		/** Enable/Disable custon Shogun's signal handler
		 * @param enable true to enable the handler, false otherwise.
		*/
		void enable_handler(bool enable)
		{
			m_active = enable;
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
