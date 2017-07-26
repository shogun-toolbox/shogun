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

#include <shogun/lib/parameter_observers/ParameterObserverCV.h>

using namespace shogun;

ParameterObserverCV::ParameterObserverCV()
{
	m_type = CROSSVALIDATION;
}

ParameterObserverCV::~ParameterObserverCV()
{
	for (auto i : m_observations)
		SG_UNREF(i)
}

void ParameterObserverCV::on_next(const shogun::TimedObservedValue& value)
{
	CHECK_OBSERVED_VALUE_TYPE(value.first.get_type());

	if (value.first.get_value().type_info().hash_code() ==
	    typeid(CrossValidationStorage*).hash_code())
	{
		CrossValidationStorage* recalled_value =
		    recall_type<CrossValidationStorage*>(value.first.get_value());
		SG_REF(recalled_value);
		m_observations.push_back(recalled_value);
	}
	else
	{
		SG_SERROR(
		    "ParameterObserverCV: The observed value received is not of "
		    "type CrossValidationStorage\n");
	}
}

void ParameterObserverCV::on_error(std::exception_ptr ptr)
{
}

void ParameterObserverCV::on_complete()
{
}

void ParameterObserverCV::clear()
{
	for (auto i : m_observations)
	{
		SG_UNREF(i)
	}
	m_observations.clear();
}

const std::vector<CrossValidationStorage*>&
ParameterObserverCV::get_observations() const
{
	return m_observations;
}
