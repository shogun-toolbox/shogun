/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */

#ifndef SHOGUN_OBSERVEDVALUE_H
#define SHOGUN_OBSERVEDVALUE_H

#include <chrono>
#include <utility>

#include <shogun/base/SGObject.h>
#include <shogun/base/some.h>

/**
 * Definitions of basic object with are needed by the Parameter
 * Observer architecture.
 */
namespace shogun
{

	template <class T>
	class ObservedValueTemplated;

	/* Timepoint */
	typedef std::chrono::steady_clock::time_point time_point;

	/**
	 * Observed value which is emitted by algorithms.
	 */
	class ObservedValue : public CSGObject
	{
	public:
		/**
		 * Constructor
		 * @param step step
		 * @param name name of the observed value
		 */
		ObservedValue(int64_t step, std::string name);

		/**
		 * Destructor
		 */
		~ObservedValue() {};

		/**
		* Helper method to generate an ObservedValue.
		* @param step the step
		* @param name the param's name we are observing
		* @param value the param's value
		* @return an ObservedValue object initialized
		*/
		template <class T>
		static Some<ObservedValue>
		make_observation(int64_t step, std::string name, T value)
		{
			return Some<ObservedValue>::from_raw(
					new ObservedValueTemplated<T>(step, name, value));
		}

		/**
	 	* Return a any version of the stored type.
	 	* @return the any value.
	 	*/
		virtual Any get_any() {
			SG_NOTIMPLEMENTED
			return make_any(nullptr);
		}

		/** @return object name */
		virtual const char* get_name() const { return "ObservedValue"; }

	protected:

		/** ObservedValue step (used by Tensorboard to print graphs) */
		int64_t m_step;
		/** Parameter's name */
		std::string m_name;
	};

	/**
	 * Observed value with a timestamp
	 */
	typedef std::pair<Some<ObservedValue>, time_point> TimedObservedValue;

	/**
	 * Helper method to convert a time_point to milliseconds
	 * @param value time point we want to convert
	 * @return the time point converted to milliseconds
	 */
	SG_FORCED_INLINE auto convert_to_millis(const time_point& value)
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(
		           value.time_since_epoch())
		    .count();
	}
}

#endif // SHOGUN_OBSERVEDVALUE_H
