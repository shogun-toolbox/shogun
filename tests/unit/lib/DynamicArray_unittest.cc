#include "sg_gtest_utilities.h"

#include <shogun/io/serialization/JsonSerializer.h>
#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/UniformIntDistribution.h>

#include "utils/Utils.h"

#include <random>

using namespace shogun;

template <typename T>
class DynamicArrayFixture : public ::testing::Test
{
protected:
	DynamicArrayFixture()
	{
	}
	virtual void SetUp()
	{
		m_array = SG_MALLOC(T, 5);
		for (int32_t i = 0; i < 5; i++)
		{
			m_array[i] = T(i);
		}
		wrapper_array = std::make_shared<DynamicArray<T>>(m_array, 5);
	}
	virtual void TearDown()
	{

	}
	virtual ~DynamicArrayFixture()
	{
	}

	std::shared_ptr<DynamicArray<T>> wrapper_array;
	T* m_array;
};

SG_TYPED_TEST_CASE(DynamicArrayFixture, sg_all_primitive_types, complex128_t);

TYPED_TEST(DynamicArrayFixture, array_ctor)
{
	EXPECT_EQ(5, this->wrapper_array->get_num_elements());
	// there is no guaranteed that a bool vector can have a capacity of 10
	// on 64 bit architecture the capacity is usually a multiple of 64
	// never trust a std::vector<bool>
	if constexpr(!std::is_same_v<TypeParam, bool>)
		EXPECT_EQ(5, this->wrapper_array->get_array_size());	
	for (index_t i = 0; i < 5; i++)
	{
		EXPECT_EQ(this->wrapper_array->get_element(i), (TypeParam)i);
	}
}

TYPED_TEST(DynamicArrayFixture, resize_array)
{
	this->wrapper_array->resize_array(10);
	// see test above
	if constexpr(!std::is_same_v<TypeParam, bool>)
		EXPECT_EQ(10, this->wrapper_array->get_array_size());
}

TYPED_TEST(DynamicArrayFixture, set_array)
{
	int32_t seed = 12;
	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	this->wrapper_array->reset_array();
	EXPECT_EQ(0, this->wrapper_array->get_num_elements());
	TypeParam* array = SG_MALLOC(TypeParam, 5);
	for (int32_t i = 0; i < 5; i++)
	{
		array[i] = (TypeParam)uniform_int_dist(prng, {1, 10});
	}
	this->wrapper_array->set_array(array, 5);

	EXPECT_EQ(5, this->wrapper_array->get_num_elements());
	for (index_t i = 0; i < 5; i++)
	{
		EXPECT_EQ(this->wrapper_array->get_element(i), array[i]);
	}
	SG_FREE(array);
}

TYPED_TEST(DynamicArrayFixture, const_set_array)
{
	int32_t seed = 12;

	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	TypeParam* array = SG_MALLOC(TypeParam, 5);
	for (int32_t i = 0; i < 5; i++)
	{
		array[i] = (TypeParam)uniform_int_dist(prng, {1, 10});
	}
	const TypeParam* const_array = array;
	this->wrapper_array->reset_array();

	// make sure array been reset
	EXPECT_EQ(0, this->wrapper_array->get_num_elements());

	this->wrapper_array->set_array(const_array, 5);
	for (index_t i = 0; i < 5; i++)
	{
		EXPECT_EQ(this->wrapper_array->get_element(i), const_array[i]);
	}
	SG_FREE(array);
}

#if 0
TYPED_TEST(DynamicArrayFixture, get_array)
{
	TypeParam* array = this->wrapper_array->get_array();

	for (index_t i = 0; i < 5; i++)
	{
		EXPECT_EQ(this->wrapper_array->get_element(i), (TypeParam)array[i]);
	}
}
#endif

TYPED_TEST(DynamicArrayFixture, push_array)
{
	this->wrapper_array->reset_array();
	EXPECT_EQ(this->wrapper_array->get_num_elements(), 0);
	this->wrapper_array->push_back((TypeParam)0);
	this->wrapper_array->push_back((TypeParam)1);
	this->wrapper_array->push_back((TypeParam)2);
	EXPECT_EQ(this->wrapper_array->get_num_elements(), 3);
	for (int32_t i = 0; i < 3; i++)
	{
		EXPECT_EQ(this->wrapper_array->get_element(i), (TypeParam)i);
	}
}

TYPED_TEST(DynamicArrayFixture, append_array)
{
	this->wrapper_array->reset_array();
	EXPECT_EQ(this->wrapper_array->get_num_elements(), 0);
	this->wrapper_array->append_element((TypeParam)0);
	this->wrapper_array->append_element((TypeParam)1);
	this->wrapper_array->append_element((TypeParam)2);
	EXPECT_EQ(this->wrapper_array->get_num_elements(), 3);
	for (int32_t i = 0; i < 3; i++)
	{
		EXPECT_EQ(this->wrapper_array->get_element(i), (TypeParam)i);
	}
}

TYPED_TEST(DynamicArrayFixture, back_operation)
{
	EXPECT_EQ(this->wrapper_array->back(), (TypeParam)4);
}

TYPED_TEST(DynamicArrayFixture, set_operation)
{
	this->wrapper_array->set_element(1, (TypeParam)4);
	EXPECT_EQ(this->wrapper_array->get_element(4), (TypeParam)1);
}

TYPED_TEST(DynamicArrayFixture, pop_operation)
{
	this->wrapper_array->pop_back();
	EXPECT_EQ(this->wrapper_array->back(), (TypeParam)3);
}

TYPED_TEST(DynamicArrayFixture, insert_operation)
{
	this->wrapper_array->insert_element((TypeParam)10, 2);
	EXPECT_EQ(this->wrapper_array->get_element(2), (TypeParam)10);
}

TYPED_TEST(DynamicArrayFixture, append_array_bool)
{
	this->wrapper_array->reset_array();
	EXPECT_EQ(this->wrapper_array->get_num_elements(), 0);
	this->wrapper_array->append_element((TypeParam)1);
	this->wrapper_array->append_element((TypeParam)0);
	this->wrapper_array->append_element((TypeParam)1);
	EXPECT_EQ(this->wrapper_array->get_num_elements(), 3);
	EXPECT_EQ(this->wrapper_array->get_element(0), (TypeParam)1);
	EXPECT_EQ(this->wrapper_array->get_element(1), (TypeParam)0);
	EXPECT_EQ(this->wrapper_array->get_element(2), (TypeParam)1);
}

TYPED_TEST(DynamicArrayFixture, save_serializable)
{
	/* generate file name */
	char filename[] = "serialization-asciiDynamicArray.XXXXXX";
	generate_temp_filename(filename);

	io::serialize(filename, this->wrapper_array, std::make_shared<io::JsonSerializer>());

	auto new_array = io::deserialize(filename, std::make_shared<io::JsonDeserializer>())->as<DynamicArray<TypeParam>>();

	ASSERT(this->wrapper_array->get_num_elements() == 5)
	for (int32_t i = 0; i < this->wrapper_array->get_num_elements(); i++)
	{
		EXPECT_EQ(
		    this->wrapper_array->get_element(i), new_array->get_element(i));
	}


	// FIXME: use fs
	unlink(filename);
}
