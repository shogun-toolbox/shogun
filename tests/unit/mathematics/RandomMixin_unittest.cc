#include <shogun/lib/common.h>
#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/mathematics/RandomMixin.h>
#include <shogun/base/some.h>
#include <gtest/gtest.h>

#include <memory>

using namespace shogun;

class MockObject : public CSGObject
{
public:
	const char* get_name() const override
	{
		return "MockObject";
	}
};

class MockNonRandom : public MockObject
{
};

class MockRandom : public RandomMixin<MockObject>
{
public:
	std::mt19937_64& m_prng = RandomMixin<MockObject>::m_prng;
};

class OneLevelNested : public MockRandom
{
public:
	OneLevelNested()
	{
		obj1 = new MockRandom();
		obj2 = nullptr;
		obj3 = new MockRandom();
		obj4 = new MockNonRandom();

		watch_param("obj1", &obj1);
		watch_param("obj2", &obj2);
		watch_param("obj3", &obj3);
		watch_param("obj4", &obj4);
	}

	~OneLevelNested()
	{
		delete obj1;
		delete obj2;
		delete obj3;
		delete obj4;
	}

	MockRandom* obj1;
	MockRandom* obj2;
	MockRandom* obj3;
	MockNonRandom* obj4;
};

class OneLevelNestedNonRandom : public MockNonRandom
{
public:
	OneLevelNestedNonRandom()
	{
		obj1 = new MockRandom();
		obj2 = nullptr;
		obj3 = new MockRandom();
		obj4 = new MockNonRandom();

		watch_param("obj1", &obj1);
		watch_param("obj2", &obj2);
		watch_param("obj3", &obj3);
		watch_param("obj4", &obj4);
	}

	~OneLevelNestedNonRandom()
	{
		delete obj1;
		delete obj2;
		delete obj3;
		delete obj4;
	}

	MockRandom* obj1;
	MockRandom* obj2;
	MockRandom* obj3;
	MockNonRandom* obj4;
};

class TwoLevelNested : public MockRandom
{
public:
	TwoLevelNested()
	{
		obj1 = new OneLevelNested();
		obj2 = new OneLevelNestedNonRandom();
		obj3 = new MockRandom();
		obj4 = new MockNonRandom();

		watch_param("obj1", &obj1);
		watch_param("obj2", &obj2);
		watch_param("obj3", &obj3);
		watch_param("obj4", &obj4);
	}

	~TwoLevelNested()
	{
		delete obj1;
		delete obj2;
		delete obj3;
		delete obj4;
	}

	OneLevelNested* obj1;
	OneLevelNestedNonRandom* obj2;
	MockRandom* obj3;
	MockNonRandom* obj4;
};

TEST(RandomMixin, basic_test)
{
	auto mock_random = std::make_unique<MockRandom>();
	mock_random->put("seed", 123);
	EXPECT_EQ(mock_random->get<int32_t>("seed"), 123);

	auto random_value = mock_random->m_prng();
	auto random_value2 = mock_random->m_prng();

	mock_random->put("seed", 345);
	EXPECT_EQ(mock_random->get<int32_t>("seed"), 345);

	auto random_value3 = mock_random->m_prng();
	auto random_value4 = mock_random->m_prng();

	mock_random->put("seed", 123);
	EXPECT_EQ(mock_random->m_prng(), random_value);
	EXPECT_EQ(mock_random->m_prng(), random_value2);

	mock_random->put("seed", 345);
	EXPECT_EQ(mock_random->m_prng(), random_value3);
	EXPECT_EQ(mock_random->m_prng(), random_value4);
}

TEST(RandomMixin, one_level_nesting_test)
{
	auto obj = std::make_unique<OneLevelNested>();
	obj->put("seed", 123);
	
	EXPECT_EQ(obj->get<int32_t>("seed"), 123);
	EXPECT_EQ(obj->obj1->get<int32_t>("seed"), 123);
	EXPECT_EQ(obj->obj3->get<int32_t>("seed"), 123);
	
	auto random_value = obj->m_prng();
	EXPECT_EQ(random_value, obj->obj1->m_prng());
	EXPECT_EQ(random_value, obj->obj3->m_prng());
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