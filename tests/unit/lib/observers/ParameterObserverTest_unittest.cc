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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2017 Giovanni De Toni
 *
 */
#include <gtest/gtest.h>

#include <shogun/lib/config.h>

#include <shogun/lib/observers/ParameterObserverLogger.h>
#include <vector>

using namespace shogun;

class MockEmitter : public SGObject
{
public:
	MockEmitter() : SGObject()
	{
	}
	virtual ~MockEmitter()
	{
	}

	virtual const char* get_name() const
	{
		return "MockEmitter";
	}

	void emit_value()
	{
		AnyParameterProperties p(
		    "Name of the observed value", ParameterProperties::READONLY);
		AnyParameterProperties p2(
		    "Name of the observed value", ParameterProperties::AUTO);
		observe<int32_t>(1, "test", 1, p);
		observe<int32_t>(1, "a", 1, p);
		observe<int32_t>(1, "b", 1, p2);
	}
};

class ParameterObserverTest : public ::testing::Test
{
protected:
	void SetUp() override
	{
		test_params = {"a", "b"};
		test_params_not_found = {"k", "j"};
		test_params_properties = {ParameterProperties::AUTO,
		                          ParameterProperties::READONLY};
	}

	void TearDown() override
	{
	}

	std::vector<std::string> test_params;
	std::vector<std::string> test_params_not_found;
	std::vector<ParameterProperties> test_params_properties;
	MockEmitter emitter;
};

TEST_F(ParameterObserverTest, filter_empty)
{
	std::shared_ptr<ParameterObserver> observer(new ParameterObserverLogger());
	emitter.subscribe(observer);
	emitter.emit_value();
	EXPECT_EQ(observer->get<int32_t>("num_observations"), 3);
	emitter.unsubscribe(observer);
}

TEST_F(ParameterObserverTest, filter_found)
{
	std::shared_ptr<ParameterObserver> observer(
	    new ParameterObserverLogger(test_params));
	emitter.subscribe(observer);
	emitter.emit_value();
	EXPECT_EQ(observer->get<int32_t>("num_observations"), 2);
	emitter.unsubscribe(observer);
}

TEST_F(ParameterObserverTest, filter_not_found)
{
	std::shared_ptr<ParameterObserver> observer(
	    new ParameterObserverLogger(test_params_not_found));
	emitter.subscribe(observer);
	emitter.emit_value();
	EXPECT_EQ(observer->get<int32_t>("num_observations"), 0);
	emitter.unsubscribe(observer);
}

TEST_F(ParameterObserverTest, filter_found_property)
{
	std::shared_ptr<ParameterObserver> observer(
	    new ParameterObserverLogger(test_params_properties));
	emitter.subscribe(observer);
	emitter.emit_value();

	EXPECT_EQ(observer->get<int32_t>("num_observations"), 3);
	emitter.unsubscribe(observer);
}

TEST_F(ParameterObserverTest, filter_not_found_property)
{
	test_params_properties.pop_back();
	std::shared_ptr<ParameterObserver> observer(
	    new ParameterObserverLogger(test_params_properties));
	emitter.subscribe(observer);
	emitter.emit_value();

	EXPECT_EQ(observer->get<int32_t>("num_observations"), 1);
	emitter.unsubscribe(observer);
}

TEST_F(ParameterObserverTest, filter_found_property_and_name)
{
	test_params_properties.pop_back();
	std::shared_ptr<ParameterObserver> observer(
	    new ParameterObserverLogger(test_params, test_params_properties));
	emitter.subscribe(observer);
	emitter.emit_value();

	EXPECT_EQ(observer->get<int32_t>("num_observations"), 1);
	emitter.unsubscribe(observer);
}
