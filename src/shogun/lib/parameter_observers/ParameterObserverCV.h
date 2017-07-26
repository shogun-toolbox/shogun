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

#ifndef SHOGUN_PARAMETEROBSERVERCV_H
#define SHOGUN_PARAMETEROBSERVERCV_H

#include <shogun/evaluation/CrossValidationStorage.h>
#include <shogun/lib/parameter_observers/ParameterObserverInterface.h>

namespace shogun
{

	/**
	 * Base ParameterObserver class for CrossValidation.
	 */
	class ParameterObserverCV : public ParameterObserverInterface
	{

	public:
		ParameterObserverCV(bool verbose = false);
		virtual ~ParameterObserverCV();

		virtual void on_next(const TimedObservedValue& value);
		virtual void on_error(std::exception_ptr ptr);
		virtual void on_complete();

		/* Erase all observations done so far */
		virtual void clear();

		/**
		 * Get vector of observations
		 * @return std::vector of observations
		 */
		const std::vector<CrossValidationStorage*>& get_observations() const;

		void print_observed_value(CrossValidationStorage* value) const;

	private:
		void print_machine_information(CMachine* machine) const;

	protected:
		/**
		 * Observation's vector
		 */
		std::vector<CrossValidationStorage*> m_observations;

		/**
		 * enable printing of information
		 */
		bool m_verbose;
	};
}

#endif // SHOGUN_PARAMETEROBSERVERCV_H
