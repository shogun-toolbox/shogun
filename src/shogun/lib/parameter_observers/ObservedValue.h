/*
* BSD 3-Clause License
*
* Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* * Neither the name of the copyright holder nor the names of its
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Written (W) 2017 Giovanni De Toni
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
		ObservedValue(int64_t step, char * name);

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
		make_observation(int64_t step, char * name, T value)
		{
			return Some<ObservedValue>::from_raw(
					new ObservedValueTemplated<T>(step, name, value));
		}

		/**
	 	* Return a any version of the stored type
	 	* @return the any value
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
		char * m_name;
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
