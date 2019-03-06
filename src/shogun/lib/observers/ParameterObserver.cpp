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

template <class T>
T* ParameterObserver::get_observation(int i)
{
	REQUIRE(i>=0 && i<this->get_num_observations(), "Observation index (%i) is out of bound.", i);
	ObservedValue v = this->m_observations[i];
	try
	{
		return &(any_cast<T>(v.get_value()));
	}
	catch (const TypeMismatchException& exc)
	{
		SG_ERROR(
				"Cannot get observation %s::%s of type %s, incompatible "
						"requested type %s.\n",
				get_name(), v.get_name(), exc.actual().c_str(),
				exc.expected().c_str());
	}
}

template <class T>
std::vector<T> ParameterObserver::get_observations(std::string name)
{
	std::vector<ObservedValue> result;
	std::vector<T> final_vector;

	// Filter the observations by keeping only the one which matches the name
	std::copy_if(m_observations.begin(), m_observations.end(), std::back_inserter(result),
	[&name](ObservedValue v){
		return (v.get_name() == name);
	});

	// If we did not find anything, the warn the user about it
	if (result.size() == 0)
	{
		SG_WARNING("%s was not found in the observation registered!", name.c_str());
	}

	// Convert the observations to the correct name
	std::transform(result.begin(), result.end(), std::back_inserter(final_vector),
	[this, &name](ObservedValue v) {
		try
		{
			return any_cast<T>(v.get_value());
		}
		catch (const TypeMismatchException& exc)
		{
			SG_ERROR(
					"Cannot get observation %s::%s of type %s, incompatible "
							"requested type %s.\n",
					this->get_name(), name.c_str(), exc.actual().c_str(),
					exc.expected().c_str());
		}
	});
	return final_vector;
}