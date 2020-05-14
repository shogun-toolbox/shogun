/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */
#include <gtest/gtest.h>

#include <shogun/lib/config.h>

#include <functional>
#include <shogun/lib/observers/ParameterObserverLogger.h>
#include <vector>

using namespace shogun;

class MockEmitter : public SGObject
{
public:
	MockEmitter() : SGObject()
	{
		this->watch_param(
		    "member_variable", &member_variable,
		    AnyParameterProperties(
		        "A member variable", ParameterProperties::NONE));
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
		AnyParameterProperties p3(
		    "Name of the observed value", ParameterProperties::NONE);

		observe<int32_t>(1, "test", 1, p);
		observe<int32_t>(1, "a", 1, p);
		observe<int32_t>(1, "b", 1, p2);
		observe<int32_t>(1, "None", 1, p3);

		std::function<int32_t()> lambda_function = [this]() {
			this->member_variable++;
			return 20;
		};

		observe<std::function<int32_t()>>(
		    1, "method", "method description", lambda_function);
	}

private:
	int32_t member_variable = 0;
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

	EXPECT_EQ(observer->get<int32_t>("num_observations"), 5);
	EXPECT_EQ(emitter.get<int32_t>("member_variable"), 1);

	emitter.unsubscribe(observer);
}

TEST_F(ParameterObserverTest, filter_found)
{
	std::shared_ptr<ParameterObserver> observer(
	    new ParameterObserverLogger(test_params));
	emitter.subscribe(observer);
	emitter.emit_value();

	EXPECT_EQ(observer->get<int32_t>("num_observations"), 2);
	EXPECT_EQ(emitter.get<int32_t>("member_variable"), 0);

	emitter.unsubscribe(observer);
}

TEST_F(ParameterObserverTest, filter_not_found)
{
	std::shared_ptr<ParameterObserver> observer(
	    new ParameterObserverLogger(test_params_not_found));
	emitter.subscribe(observer);
	emitter.emit_value();

	EXPECT_EQ(observer->get<int32_t>("num_observations"), 0);
	EXPECT_EQ(emitter.get<int32_t>("member_variable"), 0);

	emitter.unsubscribe(observer);
}

TEST_F(ParameterObserverTest, filter_found_property)
{
	std::shared_ptr<ParameterObserver> observer(
	    new ParameterObserverLogger(test_params_properties));
	emitter.subscribe(observer);
	emitter.emit_value();

	EXPECT_EQ(observer->get<int32_t>("num_observations"), 4);
	EXPECT_EQ(emitter.get<int32_t>("member_variable"), 1);

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
	EXPECT_EQ(emitter.get<int32_t>("member_variable"), 0);

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
	EXPECT_EQ(emitter.get<int32_t>("member_variable"), 0);

	emitter.unsubscribe(observer);
}

TEST_F(ParameterObserverTest, filter_all_property)
{
	std::vector<ParameterProperties> test_all_property = {
	    ParameterProperties::ALL};
	std::shared_ptr<ParameterObserver> observer(
	    new ParameterObserverLogger(test_all_property));
	emitter.subscribe(observer);
	emitter.emit_value();

	EXPECT_EQ(observer->get<int32_t>("num_observations"), 5);
	EXPECT_EQ(emitter.get<int32_t>("member_variable"), 1);

	emitter.unsubscribe(observer);
}

TEST_F(ParameterObserverTest, filter_method)
{
	std::vector<std::string> method_params = {"method"};
	std::shared_ptr<ParameterObserver> observer(
	    new ParameterObserverLogger(method_params));
	emitter.subscribe(observer);
	emitter.emit_value();

	EXPECT_EQ(observer->get<int32_t>("num_observations"), 1);
	EXPECT_EQ(observer->get_observation(0)->get_any().as<int32_t>(), 20);
	EXPECT_EQ(emitter.get<int32_t>("member_variable"), 2);

	emitter.unsubscribe(observer);
}