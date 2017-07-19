#include <gtest/gtest.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/mathematics/Math.h>

#include "utils/Utils.h"

using namespace shogun;

template <typename T>
class CDynamicArrayFixture : public ::testing::Test
{
protected:
	CDynamicArrayFixture()
	{
	}
	virtual void SetUp()
	{
		m_array = SG_MALLOC(T, 5);
		for (int32_t i = 0; i < 5; i++)
		{
			m_array[i] = (T)i;
		}
		wrapper_array = new CDynamicArray<T>(m_array, 5);
		SG_FREE(m_array);
	}
	virtual void TearDown()
	{
		SG_UNREF(wrapper_array);
	}
	virtual ~CDynamicArrayFixture()
	{
	}

	CDynamicArray<T>* wrapper_array;
	T* m_array;
};

typedef ::testing::Types<bool, char, int8_t, uint8_t, int16_t, int32_t, int64_t,
                         float32_t, float64_t>
    DynamicArrayTypes;
TYPED_TEST_CASE(CDynamicArrayFixture, DynamicArrayTypes);

TYPED_TEST(CDynamicArrayFixture, array_ctor)
{
	EXPECT_EQ(this->wrapper_array->get_num_elements(), 5);
	EXPECT_EQ(this->wrapper_array->get_array_size(), 5);
	for (index_t i = 0; i < 5; i++)
	{
		EXPECT_EQ(this->wrapper_array->get_element(i), (TypeParam)i);
	}
}

TYPED_TEST(CDynamicArrayFixture, resize_array)
{
	this->wrapper_array->resize_array(10);
	EXPECT_EQ(this->wrapper_array->get_array_size(), 15);
}

TYPED_TEST(CDynamicArrayFixture, set_array)
{
	this->wrapper_array->reset_array();
	EXPECT_EQ(this->wrapper_array->get_num_elements(), 0);
	TypeParam* array = SG_MALLOC(TypeParam, 5);
	auto prng = get_prng();
	std::uniform_int_distribution<index_t> dist(1, 10);
	for (int32_t i = 0; i < 5; i++)
	{
		array[i] = (TypeParam)dist(prng);
	}
	this->wrapper_array->set_array(array, 5);

	EXPECT_EQ(this->wrapper_array->get_num_elements(), 5);
	for (index_t i = 0; i < 5; i++)
	{
		EXPECT_EQ(this->wrapper_array->get_element(i), array[i]);
	}
	SG_FREE(array);
}

TYPED_TEST(CDynamicArrayFixture, const_set_array)
{
	TypeParam* array = SG_MALLOC(TypeParam, 5);
	auto prng = get_prng();
	std::uniform_int_distribution<index_t> dist(1, 10);
	for (int32_t i = 0; i < 5; i++)
	{
		array[i] = (TypeParam)dist(prng);
	}
	const TypeParam* const_array = array;
	this->wrapper_array->reset_array();

	// make sure array been reset
	EXPECT_EQ(this->wrapper_array->get_num_elements(), 0);

	this->wrapper_array->set_array(const_array, 5);
	for (index_t i = 0; i < 5; i++)
	{
		EXPECT_EQ(this->wrapper_array->get_element(i), const_array[i]);
	}
	SG_FREE(array);
}

TYPED_TEST(CDynamicArrayFixture, get_array)
{
	TypeParam* array = this->wrapper_array->get_array();

	for (index_t i = 0; i < 5; i++)
	{
		EXPECT_EQ(this->wrapper_array->get_element(i), (TypeParam)array[i]);
	}
}

TYPED_TEST(CDynamicArrayFixture, push_array)
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

TYPED_TEST(CDynamicArrayFixture, append_array)
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

TYPED_TEST(CDynamicArrayFixture, back_operation)
{
	EXPECT_EQ(this->wrapper_array->back(), (TypeParam)4);
}

TYPED_TEST(CDynamicArrayFixture, set_operation)
{
	this->wrapper_array->set_element(1, (TypeParam)4);
	EXPECT_EQ(this->wrapper_array->get_element(4), (TypeParam)1);
}

TYPED_TEST(CDynamicArrayFixture, pop_operation)
{
	this->wrapper_array->pop_back();
	EXPECT_EQ(this->wrapper_array->back(), (TypeParam)3);
}

TYPED_TEST(CDynamicArrayFixture, insert_operation)
{
	this->wrapper_array->insert_element((TypeParam)10, 2);
	EXPECT_EQ(this->wrapper_array->get_element(2), (TypeParam)10);
}

TYPED_TEST(CDynamicArrayFixture, append_array_bool)
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

TYPED_TEST(CDynamicArrayFixture, save_serializable)
{
	/* generate file name */
	char filename[] = "serialization-asciiCDynamicArray.XXXXXX";
	generate_temp_filename(filename);

	CSerializableAsciiFile* file = new CSerializableAsciiFile(filename, 'w');
	this->wrapper_array->save_serializable(file);
	file->close();
	SG_UNREF(file);

	file = new CSerializableAsciiFile(filename, 'r');
	CDynamicArray<TypeParam>* new_array = new CDynamicArray<TypeParam>();
	new_array->load_serializable(file);
	file->close();
	SG_UNREF(file);

	ASSERT(this->wrapper_array->get_num_elements() == 5)
	for (int32_t i = 0; i < this->wrapper_array->get_num_elements(); i++)
	{
		EXPECT_EQ(
		    this->wrapper_array->get_element(i), new_array->get_element(i));
	}

	SG_UNREF(new_array);
	unlink(filename);
}
