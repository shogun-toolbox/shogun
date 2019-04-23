#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/mathematics/RandomMixin.h>
#include <gtest/gtest.h>
#include "../base/MockObject.h"

#include <memory>
#include <array>

using namespace shogun;

class MockNonRandom : public MockObject
{
};

class MockRandom : public RandomMixin<MockObject>
{
public:
	auto sample() -> decltype(m_prng())
	{
		return m_prng();
	}
};

class OneLevelNested : public MockRandom
{
public:
	OneLevelNested()
	{
		obj1 = std::make_shared<MockRandom>();
		obj2 = nullptr;
		obj3 = std::make_shared<MockRandom>();
		obj4 = std::make_shared<MockNonRandom>();

		watch_param("obj1", &obj1);
		watch_param("obj2", &obj2);
		watch_param("obj3", &obj3);
		watch_param("obj4", &obj4);
	}

	~OneLevelNested()
	{
	}

	std::shared_ptr<MockRandom> obj1;
	std::shared_ptr<MockRandom> obj2;
	std::shared_ptr<MockRandom> obj3;
	std::shared_ptr<MockNonRandom> obj4;
};

class OneLevelNestedNonRandom : public MockNonRandom
{
public:
	OneLevelNestedNonRandom()
	{
		obj1 = std::make_shared<MockRandom>();
		obj2 = nullptr;
		obj3 = std::make_shared<MockRandom>();
		obj4 = std::make_shared<MockNonRandom>();

		watch_param("obj1", &obj1);
		watch_param("obj2", &obj2);
		watch_param("obj3", &obj3);
		watch_param("obj4", &obj4);
	}

	~OneLevelNestedNonRandom()
	{
	}

	std::shared_ptr<MockRandom> obj1;
	std::shared_ptr<MockRandom> obj2;
	std::shared_ptr<MockRandom> obj3;
	std::shared_ptr<MockNonRandom> obj4;
};

class TwoLevelNested : public MockRandom
{
public:
	TwoLevelNested()
	{
		obj1 = std::make_shared<OneLevelNested>();
		obj2 = std::make_shared<OneLevelNestedNonRandom>();
		obj3 = std::make_shared<MockRandom>();
		obj4 = std::make_shared<MockNonRandom>();

		watch_param("obj1", &obj1);
		watch_param("obj2", &obj2);
		watch_param("obj3", &obj3);
		watch_param("obj4", &obj4);
	}

	~TwoLevelNested()
	{
	}

	std::shared_ptr<OneLevelNested> obj1;
	std::shared_ptr<OneLevelNestedNonRandom> obj2;
	std::shared_ptr<MockRandom> obj3;
	std::shared_ptr<MockNonRandom> obj4;
};

TEST(RandomMixin, reproducibility_test)
{
	auto mock_random = std::make_unique<MockRandom>();
	mock_random->put("seed", 123);
	EXPECT_EQ(mock_random->get<int32_t>("seed"), 123);

	std::array<decltype(mock_random->sample()), 100> values1;
	for(auto& val : values1)
		val = mock_random->sample();

	mock_random->put("seed", 345);
	EXPECT_EQ(mock_random->get<int32_t>("seed"), 345);

	std::array<decltype(mock_random->sample()), 100> values2;
	for(auto& val : values2)
		val = mock_random->sample();

	mock_random->put("seed", 123);
	for(auto& val : values1)
		EXPECT_EQ(mock_random->sample(), val);

	mock_random->put("seed", 345);
	for(auto& val : values2)
		EXPECT_EQ(mock_random->sample(), val);
}

TEST(RandomMixin, one_level_nesting_test)
{
	auto obj = std::make_unique<OneLevelNested>();
	obj->put("seed", 123);
	
	EXPECT_EQ(obj->get<int32_t>("seed"), 123);
	EXPECT_EQ(obj->obj1->get<int32_t>("seed"), 123);
	EXPECT_EQ(obj->obj3->get<int32_t>("seed"), 123);
	
	auto random_value = obj->sample();
	EXPECT_EQ(random_value, obj->obj1->sample());
	EXPECT_EQ(random_value, obj->obj3->sample());
}

TEST(RandomMixin, two_level_nesting_test)
{
	auto obj = std::make_unique<TwoLevelNested>();
	obj->put("seed", 123);

	EXPECT_EQ(obj->get<int32_t>("seed"), 123);
	EXPECT_EQ(obj->obj1->get<int32_t>("seed"), 123);
	EXPECT_EQ(obj->obj3->get<int32_t>("seed"), 123);

	EXPECT_EQ(obj->obj1->get<int32_t>("seed"), 123);
	EXPECT_EQ(obj->obj1->obj1->get<int32_t>("seed"), 123);
	EXPECT_EQ(obj->obj1->obj3->get<int32_t>("seed"), 123);
	
	EXPECT_EQ(obj->obj2->obj1->get<int32_t>("seed"), 123);
	EXPECT_EQ(obj->obj2->obj3->get<int32_t>("seed"), 123);
}
