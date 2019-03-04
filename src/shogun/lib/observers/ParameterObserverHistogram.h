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
#include <shogun/lib/config.h>
#ifdef HAVE_TFLOGGER

#ifndef SHOGUN_PARAMETEROBSERVERHISTOGRAM_H
#define SHOGUN_PARAMETEROBSERVERHISTOGRAM_H

#include <shogun/base/SGObject.h>
#include <shogun/lib/observers/ParameterObserverTensorBoard.h>

namespace shogun
{
	/**
	 * Implementation of a ParameterObserver which write to file
	 * histograms, given object emitted from a parameter observable.
	 */
	class ParameterObserverHistogram : public ParameterObserverTensorBoard
	{

	public:
		ParameterObserverHistogram();
		ParameterObserverHistogram(std::vector<std::string>& parameters);
		ParameterObserverHistogram(
		    const std::string& filename, std::vector<std::string>& parameters);
		~ParameterObserverHistogram();

		virtual void on_error(std::exception_ptr);
		virtual void on_complete();

		/**
		* Get class name.
		* @return class name
		*/
		virtual const char* get_name() const
		{
			return "ParameterObserverHistogram";
		}

	protected:
		virtual void on_next_impl(const TimedObservedValue& value);
	};
}

#endif // SHOGUN_PARAMETEROBSERVERHISTOGRAM_H
#endif // HAVE_TFLOGGER
