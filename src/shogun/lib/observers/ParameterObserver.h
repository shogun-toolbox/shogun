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
#ifndef SHOGUN_PARAMETEROBSERVER_H
#define SHOGUN_PARAMETEROBSERVER_H

#include <stdexcept>
#include <vector>

#include <shogun/base/SGObject.h>
#include <shogun/lib/any.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/lib/observers/observers_utils.h>

namespace shogun
{

	/**
	 * Interface for the parameter observer classes
	 */
	class ParameterObserver : public SGObject
	{

	public:
		/**
		* Default constructor
		*/
		ParameterObserver();

		/**
		 * Constructor
		 * @param parameters list of parameters which we want to watch over
		 */
		ParameterObserver(std::vector<std::string>& parameters);

		/**
		 * Constructor
		 * @param filename name of the generated output file
		 * @param parameters list of parameters which we want to watch over
		 */
		ParameterObserver(
		    const std::string& filename, std::vector<std::string>& parameters);
		/**
		 * Virtual destructor
		 */
		virtual ~ParameterObserver();

		/**
		 * Filter function, check if the parameter name supplied is what
		 * we want to monitor
		 * @param param the param name
		 * @return true if param is found inside of m_parameters list
		 */
		virtual bool filter(const std::string& param);

		/**
		 * Return a single observation from the received ones (not SG_REF).
		 * @tparam T the type of the observation
		 * @param i the index
		 * @return the observation casted to the requested type
		 */
		ObservedValue* get_observation(index_t i)
		{
			REQUIRE(
			    i >= 0 && i < this->get_num_observations(),
			    "Observation index (%i) is out of bound (total observations "
			    "%i)",
			    i, this->get_num_observations());
			return this->m_observations[i].get();
		};

		/**
		 * Erase all observations registered so far by the observer.
		 */
		virtual void clear()
		{
			m_observations.clear();
		};

		/**
		 * Method which will be called when the parameter observable emits a
		 * value.
		 * @param value the value emitted by the parameter observable
		 */
		void on_next(const TimedObservedValue& value)
		{
			m_observations.push_back(value.first);
			on_next_impl(value);
		};

		/**
		 * Method which will be called on errors
		 */
		virtual void on_error(std::exception_ptr) = 0;
		/**
		 * Method which will be called on completion
		 */
		virtual void on_complete() = 0;

		/**
		 * Get the name of this class
		 * @return name as a string
		 */
		virtual const char* get_name() const
		{
			return "ParameterObserver";
		}

	protected:
		/**
		 * Get the total number of observation received.
		 * @return number of obsevation received.
		 */
		index_t get_num_observations() const;

		/**
		 * Implementation of the on_next method which will be needed
		 * in order to process the observed value
		 * @param value the observed value
		 */
		virtual void on_next_impl(const TimedObservedValue& value) = 0;

		/**
		 * List of parameter's names we want to monitor
		 */
		std::vector<std::string> m_observed_parameters;

		/**
		 * Observations recorded each time we compute on_next()
		 */
		std::vector<std::shared_ptr<ObservedValue>> m_observations;

		/**
		 * Subscription id set when I subscribe to a machine
		 */
		int64_t m_subscription_id;
	};
}

#endif // SHOGUN_PARAMETEROBSERVER_H
