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

#include "sg_gtest_utilities.h"

#include <shogun/io/TBOutputFormat.h>
#include <shogun/lib/any.h>
#include <shogun/lib/observers/ObservedValueTemplated.h>
#include <shogun/lib/observers/observers_utils.h>
#include <shogun/lib/tfhistogram/histogram.h>
#include <tflogger/event.pb.h>
#include <tflogger/summary.pb.h>
#include <utility>
#include <vector>

using namespace shogun;

template <class T>
void test_case_scalar(T value_val)
{
	T v = value_val;
	tensorflow::Event event_ex;
	auto summary = event_ex.mutable_summary();
	auto summaryValue = summary->add_value();
	summaryValue->set_tag("test");
	summaryValue->set_node_name("node");
	summaryValue->set_simple_value(v);
	TBOutputFormat tmp;

	time_point timestamp;
	auto emitted_value = std::shared_ptr<ObservedValue>(
	    new ObservedValueTemplated<T>(1, "test", "test description", v));

	std::string node_name = "node";
	auto event_gen =
	    tmp.convert_scalar(std::make_pair(emitted_value, timestamp), node_name);
	EXPECT_EQ(event_gen.summary().value(0).simple_value(), v);
	EXPECT_EQ(event_gen.summary().value(0).tag(), "test");
	EXPECT_EQ(event_gen.summary().value(0).node_name(), "node");
}

template <class T>
void test_case_scalar_error(T value_val)
{
	T v = value_val;
	TBOutputFormat tmp;

	time_point timestamp;
	auto emitted_value = std::shared_ptr<ObservedValue>(
	    new ObservedValueTemplated<T>(1, "test", "test description", v));

	std::string node_name = "node";
	EXPECT_THROW(
	    tmp.convert_scalar(std::make_pair(emitted_value, timestamp), node_name),
	    ShogunException);
}

template <class T>
void test_case_vector(std::vector<T> v)
{
	tensorflow::Event event_ex;
	auto summary = event_ex.mutable_summary();
	auto summaryValue = summary->add_value();
	summaryValue->set_tag("test");
	summaryValue->set_node_name("node");

	tensorflow::histogram::Histogram h;
	tensorflow::HistogramProto* hp = new tensorflow::HistogramProto();
	for (auto value_v : v)
		h.Add(value_v);
	h.EncodeToProto(hp, true);
	summaryValue->set_allocated_histo(hp);

	TBOutputFormat tmp;

	time_point timestamp;
	auto emitted_value = std::shared_ptr<ObservedValue>(
	    new ObservedValueTemplated<std::vector<T>>(
	        1, "test", "test description", v));

	std::string node_name = "node";
	auto event_gen =
	    tmp.convert_vector(std::make_pair(emitted_value, timestamp), node_name);

	tensorflow::histogram::Histogram h2;
	h2.DecodeFromProto(event_gen.summary().value(0).histo());

	EXPECT_EQ(h2.ToString(), h.ToString());
	EXPECT_EQ(event_gen.summary().value(0).tag(), "test");
	EXPECT_EQ(event_gen.summary().value(0).node_name(), "node");
}

template <class T>
void test_case_vector_error(std::vector<T> v)
{
	TBOutputFormat tmp;

	time_point timestamp;
	auto emitted_value = std::shared_ptr<ObservedValue>(
	    new ObservedValueTemplated<std::vector<T>>(
	        1, "test", "test_description", v));

	std::string node_name = "node";
	EXPECT_THROW(
	    tmp.convert_vector(std::make_pair(emitted_value, timestamp), node_name),
	    ShogunException);
}

template <typename T>
class TBOutputFormatTest : public ::testing::Test
{
};

SG_TYPED_TEST_CASE(TBOutputFormatTest, sg_all_primitive_types, bool, complex128_t);

TYPED_TEST(TBOutputFormatTest, convert_all_types_scalar)
{
	test_case_scalar<TypeParam>(1);
};

TEST(TBOutputFormatTest, fail_convert_scalar)
{
	test_case_scalar_error<complex128_t>(1);
};

TYPED_TEST(TBOutputFormatTest, convert_all_types_histo)
{
	std::vector<TypeParam> v;
	v.push_back((TypeParam)1);
	v.push_back((TypeParam)2);
	test_case_vector<TypeParam>(v);
};

TEST(TBOutputFormat, fail_convert_histo)
{
	std::vector<complex128_t> v;
	v.push_back((complex128_t)1);
	v.push_back((complex128_t)2);
	test_case_vector_error<complex128_t>(v);
}

#endif // HAVE_TFLOGGER
