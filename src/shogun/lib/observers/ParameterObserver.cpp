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
#include <shogun/lib/RefCount.h>
#include <shogun/lib/observers/ParameterObserver.h>

using namespace shogun;

ParameterObserver::ParameterObserver() : m_parameters()
{
}

ParameterObserver::ParameterObserver(std::vector<std::string>& parameters)
    : m_parameters(parameters)
{
}

ParameterObserver::ParameterObserver(
    const std::string& filename, std::vector<std::string>& parameters)
    : m_parameters(parameters)
{
}

ParameterObserver::~ParameterObserver()
{
}

bool ParameterObserver::filter(const std::string& param)
{
	// If there are no specified parameters, then watch everything
	if (m_parameters.size() == 0)
		return true;

	for (auto v : m_parameters)
		if (v == param)
			return true;
	return false;
}

Any ParameterObserver::get_observation(int i)
{
	REQUIRE(
	    i < m_observations.size(), "The id of the observation is to great.");
	REQUIRE(i >= 0, "The id of the observation must be greater than zero.");
	return m_observations[i];
}