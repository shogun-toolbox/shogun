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
#include <shogun/lib/common.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/lib/tfhistogram/histogram.h>
#include <shogun/lib/type_case.h>
#include <vector>

using namespace shogun;

#define CHECK_TYPE_HISTO(type)                                                 \
	else if (                                                                  \
	    value.first->get_any().type_info().hash_code() ==                      \
	    typeid(type).hash_code())                                              \
	{                                                                          \
		tensorflow::histogram::Histogram h;                                    \
		tensorflow::HistogramProto* hp = new tensorflow::HistogramProto();     \
		auto v = any_cast<type>(value.first->get_any());                       \
		for (auto value_v : v)                                                 \
			h.Add(value_v);                                                    \
		h.EncodeToProto(hp, true);                                             \
		summaryValue->set_allocated_histo(hp);                                 \
	}

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

	if (value.first->get_any().type_info().hash_code() ==
	    typeid(std::vector<int8_t>).hash_code())
	{
		tensorflow::histogram::Histogram h;
		tensorflow::HistogramProto* hp = new tensorflow::HistogramProto();
		auto v = any_cast<std::vector<int8_t>>(value.first->get_any());
		for (auto value_v : v)
			h.Add(value_v);
		h.EncodeToProto(hp, true);
		summaryValue->set_allocated_histo(hp);
	}
	CHECK_TYPE_HISTO(std::vector<uint8_t>)
	CHECK_TYPE_HISTO(std::vector<int16_t>)
	CHECK_TYPE_HISTO(std::vector<uint16_t>)
	CHECK_TYPE_HISTO(std::vector<int32_t>)
	CHECK_TYPE_HISTO(std::vector<uint32_t>)
	CHECK_TYPE_HISTO(std::vector<int64_t>)
	CHECK_TYPE_HISTO(std::vector<uint64_t>)
	CHECK_TYPE_HISTO(std::vector<float32_t>)
	CHECK_TYPE_HISTO(std::vector<float64_t>)
	CHECK_TYPE_HISTO(std::vector<floatmax_t>)
	CHECK_TYPE_HISTO(std::vector<char>)
	else
	{
		SG_ERROR(
		    "Unsupported type %s", value.first->get_any().type_info().name());
	}

	return e;
}

#endif // HAVE_TFLOGGER
