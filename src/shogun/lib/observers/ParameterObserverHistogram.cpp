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

#include <shogun/io/TBOutputFormat.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/lib/observers/ParameterObserverHistogram.h>

using namespace shogun;

CParameterObserverHistogram::CParameterObserverHistogram()
    : ParameterObserverTensorBoard()
{
}

CParameterObserverHistogram::CParameterObserverHistogram(
    std::vector<std::string>& parameters)
    : ParameterObserverTensorBoard(parameters)
{
}

CParameterObserverHistogram::CParameterObserverHistogram(
    const std::string& filename, std::vector<std::string>& parameters)
    : ParameterObserverTensorBoard(filename, parameters)
{
}

CParameterObserverHistogram::~CParameterObserverHistogram()
{
}

void CParameterObserverHistogram::on_next_impl(const TimedObservedValue& value)
{
	auto node_name = std::string("node");
	auto format = TBOutputFormat();
	auto event_value = format.convert_vector(value, node_name);
	m_writer.writeEvent(event_value);
}

void CParameterObserverHistogram::on_error(std::exception_ptr)
{
}

void CParameterObserverHistogram::on_complete()
{
}

#endif // HAVE_TFLOGGER
