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

#include <chrono>
#include <shogun/io/TBOutputFormat.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/lib/tfhistogram/histogram.h>
#include <shogun/lib/type_case.h>

using namespace shogun;

TBOutputFormat::TBOutputFormat(){};

TBOutputFormat::~TBOutputFormat(){};

tensorflow::Event TBOutputFormat::convert_scalar(
    const TimedObservedValue& value, std::string& node_name)
{
	tensorflow::Event e;
	std::time_t now_t = convert_to_millis(value.second);
	e.set_wall_time(now_t);
	e.set_step(value.first->get<int64_t>("step"));

	tensorflow::Summary* summary = e.mutable_summary();
	auto summaryValue = summary->add_value();
	summaryValue->set_tag(value.first->get<std::string>("name"));
	summaryValue->set_node_name(node_name);

	auto write_summary = [&summaryValue=summaryValue](auto val) {
		summaryValue->set_simple_value(val);
	};

	sg_any_dispatch(value.first->get_any(), sg_all_typemap, write_summary);

	return e;
}

tensorflow::Event TBOutputFormat::convert_vector(
    const TimedObservedValue& value, std::string& node_name)
{
	tensorflow::Event e;
	std::time_t now_t = convert_to_millis(value.second);
	e.set_wall_time(now_t);
	e.set_step(value.first->get<int64_t>("step"));

	tensorflow::Summary* summary = e.mutable_summary();
	auto summaryValue = summary->add_value();
	summaryValue->set_tag(value.first->get<std::string>("name"));
	summaryValue->set_node_name(node_name);

	tensorflow::histogram::Histogram h;
	tensorflow::HistogramProto* hp = new tensorflow::HistogramProto();

	auto write_summary = [&h](auto val) {
		for (auto value_v : val)
			h.Add(value_v);
	};

	sg_any_dispatch(value.first->get_any(), sg_all_typemap, None{}, write_summary);

	h.EncodeToProto(hp, true);
	summaryValue->set_allocated_histo(hp);

	return e;
}

#endif // HAVE_TFLOGGER
