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
#ifndef SHOGUN_PARAMETEROBSERVERINTERFACE_H
#define SHOGUN_PARAMETEROBSERVERINTERFACE_H

#include <stdexcept>
#include <utility>
#include <vector>

#include <shogun/lib/any.h>

namespace shogun
{
	/**
	 * Interface for the parameter observer classes
	 */
	class ParameterObserverInterface
	{

	public:
		/* One observed value, composed of:
		*  - step (for the graph x axis);
		*  - a pair composed of: parameter's name + parameter's value
		*/
		typedef std::pair<int64_t, std::pair<std::string, Any>> ObservedValue;

		/**
		* Default constructor
		*/
		ParameterObserverInterface();

		/**
		 * Constructor
		 * @param parameters list of parameters which we want to watch over
		 */
		ParameterObserverInterface(std::vector<std::string>& parameters);

		/**
		 * Constructor
		 * @param filename name of the generated output file
		 * @param parameters list of parameters which we want to watch over
		 */
		ParameterObserverInterface(
		    const std::string& filename, std::vector<std::string>& parameters);
		/**
		 * Virtual destructor
		 */
		virtual ~ParameterObserverInterface();

		/**
		 * Filter function, check if the parameter name supplied is what
		 * we want to monitor
		 * @param param the param name
		 * @return true if param is found inside of m_parameters list
		 */
		virtual bool filter(const std::string& param) = 0;

		/**
		 * Method which will be called when the parameter observable emits a
		 * value.
		 * @param value the value emitted by the parameter observable
		 */
		virtual void on_next(const ObservedValue& value) = 0;
		/**
		 * Method which will be called on errors
		 */
		virtual void on_error(std::exception_ptr) = 0;
		/**
		 * Method which will be called on completion
		 */
		virtual void on_complete() = 0;

	protected:
		/**
		 * List of parameter's names we want to monitor
		 */
		std::vector<std::string> m_parameters;
	};
}

#endif // SHOGUN_PARAMETEROBSERVER_H
