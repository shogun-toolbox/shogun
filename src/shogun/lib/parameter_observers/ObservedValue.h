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
#include <shogun/lib/any.h>
#include <shogun/lib/common.h>
#include <utility>

/**
 * Definitions of basic object with are needed by the Parameter
 * Observer architecture.
 */
namespace shogun
{
	/* Timepoint */
	typedef std::chrono::steady_clock::time_point time_point;

	/* Type of the observed value */
	enum SG_OBS_VALUE_TYPE
	{
		TENSORBOARD,
		CROSSVALIDATION
	};

	/**
	 * Observed value which is emitted by algorithms.
	 */
	class ObservedValue
	{
	public:
		/**
		 * Constructor
		 * @param step step
		 * @param name param's name
		 * @param value Any-wrapped value of the param
		 */
		ObservedValue(
		    int64_t step, std::string& name, Any value, SG_OBS_VALUE_TYPE type)
		    : m_step(step), m_name(name), m_value(value), m_type(type)
		{
		}

		~ObservedValue(){};

		/**
		 * Get the step
		 * @return an integer representing the step
		 */
		int64_t get_step() const
		{
			return m_step;
		}

		/**
		 * Set the step
		 * @param step step
		 */
		void set_step(int64_t step)
		{
			m_step = step;
		}

		/**
		 * Get the param's name
		 * @return param's name
		 */
		const std::string& get_name() const
		{
			return m_name;
		}

		/**
		 * Set the param's name
		 * @param name
		 */
		void set_name(const std::string& name)
		{
			m_name = name;
		}

		/**
		 * Get the Any-wrapped value
		 * @return Any-wrapped value
		 */
		const Any& get_value() const
		{
			return m_value;
		}

		/**
		 * Set the param's value
		 * @param value
		 */
		void set_value(const Any& value)
		{
			m_value = value;
		}

		/**
		 * Get the type of this ObservedValue
		 * @return observed value type
		 */
		SG_OBS_VALUE_TYPE get_type() const
		{
			return m_type;
		}

		/**
		 * Set the observed value type
		 * @param type type of this observed value
		 */
		void set_type(const SG_OBS_VALUE_TYPE type)
		{
			m_type = type;
		}

		/**
		* Helper method to generate an ObservedValue (TensorBoard oriented)
		* @param step the step
		* @param name the param's name we are observing
		* @param value the param's value
		* @return an ObservedValue object initialized
		*/
		static ObservedValue
		make_observation(int64_t step, std::string& name, Any value)
		{
			return ObservedValue(step, name, value, TENSORBOARD);
		}

	protected:
		/** ObservedValue step (used by Tensorboard to print graphs) */
		int64_t m_step;
		/** Parameter's name */
		std::string m_name;
		/** Parameter's value */
		Any m_value;
		/** ObservedValue type */
		SG_OBS_VALUE_TYPE m_type;
	};

	/**
	 * Observed value with a timestamp
	 */
	typedef std::pair<ObservedValue, time_point> TimedObservedValue;

	/**
	 * Helper method to convert a time_point to milliseconds
	 * @param value time point we want to convert
	 * @return the time point converted to milliseconds
	 */
	SG_FORCED_INLINE double convert_to_millis(const time_point& value)
	{
		return std::chrono::duration_cast<std::chrono::milliseconds>(
		           value.time_since_epoch())
		    .count();
	}
}

#endif // SHOGUN_OBSERVEDVALUE_H
