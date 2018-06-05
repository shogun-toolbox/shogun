/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shubham Shukla
 */

#ifndef __STOPPABLESGOBJECT_H_
#define __STOPPABLESGOBJECT_H_

#include <shogun/base/SGObject.h>
#include <shogun/base/init.h>

#include <condition_variable>
#include <mutex>

namespace shogun
{
#define COMPUTATION_CONTROLLERS                                                \
	if (cancel_computation())                                                  \
		break;                                                                 \
	pause_computation();

	/**
	 * Class that abstracts all premature stopping code
	 */
	class CStoppableSGObject : public CSGObject
	{
	public:
		/** constructor */
		CStoppableSGObject();

		/** destructor */
		virtual ~CStoppableSGObject();

#ifndef SWIG
		/** @return whether the algorithm needs to be stopped */
		SG_FORCED_INLINE bool cancel_computation() const
		{
			/* Execute the callback, if present*/
			return (m_callback) ? (m_cancel_computation.load() || m_callback())
			                    : m_cancel_computation.load();
		}
#endif

#ifndef SWIG
		/** Pause the algorithm if the flag is set */
		SG_FORCED_INLINE void pause_computation()
		{
			if (m_pause_computation_flag.load())
			{
				std::unique_lock<std::mutex> lck(m_mutex);
				while (m_pause_computation_flag.load())
					m_pause_computation.wait(lck);
			}
		}
#endif

#ifndef SWIG
		/** Resume current computation (sets the flag) */
		SG_FORCED_INLINE void resume_computation()
		{
			std::unique_lock<std::mutex> lck(m_mutex);
			m_pause_computation_flag = false;
			m_pause_computation.notify_all();
		}
#endif

		/**
		 * Set an additional stopping condition
		 * @param callback method that implements an additional stopping
		 * condition
		 */
		void set_callback(std::function<bool()> callback);

		virtual const char* get_name() const
		{
			return "StoppableSGObject";
		}

	protected:
		/** connect the machine instance to the signal handler */
		rxcpp::subscription connect_to_signal_handler();

		/** reset the computation variables */
		void reset_computation_variables();

		/** sets cancel computation flag */
		void on_next();

		/** The action which will be done when the user decides to
		* premature stop the CMachine execution */
		virtual void on_next_impl();

		/** sets pause computation flag and resumes after action is complete */
		void on_pause();

		/** The action which will be done when the user decides to
		* pause the CMachine execution */
		virtual void on_pause_impl();

		/** These actions which will be done when the user decides to
		* return to prompt and terminate the program execution */
		void on_complete();
		virtual void on_complete_impl();

	protected:
		/** Cancel computation */
		std::atomic<bool> m_cancel_computation;

		/** Pause computation flag */
		std::atomic<bool> m_pause_computation_flag;

		/** Conditional variable to make threads wait */
		std::condition_variable m_pause_computation;

		/** Mutex used to pause threads */
		std::mutex m_mutex;

		std::function<bool(void)> m_callback;
	};
}
#endif
