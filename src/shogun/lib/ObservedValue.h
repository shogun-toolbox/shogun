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
#include <utility>

/**
 * Definitions of basic object with are needed by the Parameter
 * Observer architecture.
 */
namespace shogun
{
	/* Chrono timepoint */
	typedef std::chrono::time_point<
	    std::chrono::_V2::steady_clock,
	    std::chrono::duration<long int, std::ratio<1l, 1000000000l>>>
	    time_point;

	/* One observed value, composed of:
	 *  - step (for the graph x axis);
	 *  - parameter's name;
	 *  - parameter's value (Any wrapped);
	 */
	struct ObservedValue
	{
		int64_t step;
		std::string name;
		Any value;
	};

	/**
	 * Observed value with a timestamp
	 */
	typedef std::pair<ObservedValue, time_point> TimedObservedValue;

	/**
	 * Helper method to convert a time_point to std::time_t
	 * @param value time point we want to convert
	 * @return the time point converted to std::time_t
	 */
	inline std::time_t convert_to_time_t(const time_point& value)
	{
		return std::chrono::system_clock::to_time_t(
		    std::chrono::system_clock::now() +
		    (value - std::chrono::steady_clock::now()));
	}
}

#endif // SHOGUN_OBSERVEDVALUE_H
