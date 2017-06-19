#include <gtest/gtest.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/mathematics/Math.h>

#include "utils/Utils.h"

using namespace shogun;

template <typename T>
class CDynamicArrayTest : public ::testing::Test
{
protected:
	CDynamicArrayTest()
	{
	}
	virtual void SetUp()
	{
		default_array = new CDynamicArray<T>();
		custom_array = new CDynamicArray<T>(5);
	}
	virtual void TearDown()
	{
		SG_FREE(default_array);
		SG_FREE(custom_array);
	}
	virtual ~CDynamicArrayTest()
	{
	}
	CDynamicArray<T>* default_array;
	CDynamicArray<T>* custom_array;
};

class CDynamicArrayFixture : public ::testing::Test
{
protected:
	CDynamicArrayFixture()
	{
		b_array = SG_MALLOC(bool, 5);
		i_array = SG_MALLOC(int32_t, 5);
		for (int32_t i = 0; i < 5; i++)
		{
			b_array[i] = true;
		}
		for (int32_t i = 0; i < 5; i++)
		{
			i_array[i] = i;
		}
	}
	virtual void SetUp()
	{
		wrapper_array_b = new CDynamicArray<bool>(this->b_array, 5);
		wrapper_array_i = new CDynamicArray<int32_t>(this->i_array, 5);
	}
	virtual void TearDown()
	{
		SG_FREE(wrapper_array_b);
		SG_FREE(wrapper_array_i);
	}
	virtual ~CDynamicArrayFixture()
	{
		SG_FREE(b_array);
		SG_FREE(i_array);
	}
	CDynamicArray<bool>* wrapper_array_b;
	CDynamicArray<int32_t>* wrapper_array_i;

	bool* b_array;
	int32_t* i_array;
};

typedef ::testing::Types<int32_t, bool> Implementations;

TYPED_TEST_CASE(CDynamicArrayTest, Implementations);

TYPED_TEST(CDynamicArrayTest, default_array_ctor)
{
	EXPECT_EQ(this->default_array->get_num_elements(), 0);
	EXPECT_EQ(this->default_array->get_array_size(), 128);
}

TYPED_TEST(CDynamicArrayTest, custom_array_ctor)
{
	EXPECT_EQ(this->custom_array->get_num_elements(), 0);
	EXPECT_EQ(this->custom_array->get_array_size(), 5);
}

TEST_F(CDynamicArrayFixture, wrapper_array_ctor)
{
	EXPECT_EQ(wrapper_array_b->get_num_elements(), 5);
	EXPECT_EQ(wrapper_array_i->get_num_elements(), 5);
	for (index_t i = 0; i < 5; i++)
	{
		EXPECT_EQ(wrapper_array_b->get_element(i), true);
		EXPECT_EQ(wrapper_array_i->get_element(i), i);
	}
}

TYPED_TEST(CDynamicArrayTest, resize_array)
{
	this->default_array->resize_array(10);
	this->custom_array->resize_array(10);
	EXPECT_EQ(this->default_array->get_array_size(), 128);
	EXPECT_EQ(this->custom_array->get_array_size(), 15);
}

TEST_F(CDynamicArrayFixture, set_array)
{
	bool* b_array_n = SG_MALLOC(bool, 5);
	int32_t* i_array_n = SG_MALLOC(int32_t, 5);
	for (int32_t i = 0; i < 5; i++)
	{
		i_array_n[i] = CMath::random(1, 10);
	}
	for (int32_t i = 0; i < 5; i++)
	{
		b_array_n[i] = CMath::random(0, 1);
	}
	wrapper_array_i->set_array(i_array_n, 5, 5);
	wrapper_array_b->set_array(b_array_n, 5, 5);

	EXPECT_EQ(wrapper_array_i->get_num_elements(), 5);
	for (index_t i = 0; i < 5; i++)
	{
		EXPECT_EQ(wrapper_array_i->get_element(i), i_array_n[i]);
		EXPECT_EQ(wrapper_array_b->get_element(i), b_array_n[i]);
	}

	SG_FREE(b_array_n);
	SG_FREE(i_array_n);
}

TEST_F(CDynamicArrayFixture, const_set_array)
{
	const bool* b_array_n = b_array;
	const int32_t* i_array_n = i_array;

	wrapper_array_i->reset_array();
	wrapper_array_b->reset_array();

	// make sure array been reset
	EXPECT_EQ(wrapper_array_i->get_num_elements(), 0);
	EXPECT_EQ(wrapper_array_b->get_num_elements(), 0);

	wrapper_array_i->set_array(i_array_n, 5);
	wrapper_array_b->set_array(b_array_n, 5);
	for (index_t i = 0; i < 5; i++)
	{
		EXPECT_EQ(wrapper_array_i->get_element(i), i_array_n[i]);
		EXPECT_EQ(wrapper_array_b->get_element(i), b_array_n[i]);
	}
}

TEST_F(CDynamicArrayFixture, get_array)
{
	bool* b_array_n = wrapper_array_b->get_array();
	int32_t* i_array_n = wrapper_array_i->get_array();

	for (index_t i = 0; i < 5; i++)
	{
		EXPECT_EQ(wrapper_array_i->get_element(i), i_array_n[i]);
		EXPECT_EQ(wrapper_array_b->get_element(i), b_array_n[i]);
	}

	delete b_array_n;
	delete i_array_n;
}

TEST_F(CDynamicArrayFixture, push_array_int)
{
	wrapper_array_i->reset_array();
	EXPECT_EQ(wrapper_array_i->get_num_elements(), 0);
	wrapper_array_i->push_back(0);
	wrapper_array_i->push_back(1);
	wrapper_array_i->push_back(2);
	EXPECT_EQ(wrapper_array_i->get_num_elements(), 3);
	for (int32_t i = 0; i < 3; i++)
	{
		EXPECT_EQ(wrapper_array_i->get_element(i), i);
	}
}

TEST_F(CDynamicArrayFixture, push_array_bool)
{
	wrapper_array_b->reset_array();
	EXPECT_EQ(wrapper_array_b->get_num_elements(), 0);
	wrapper_array_b->push_back(true);
	wrapper_array_b->push_back(false);
	wrapper_array_b->push_back(true);
	EXPECT_EQ(wrapper_array_b->get_num_elements(), 3);
	EXPECT_EQ(wrapper_array_b->get_element(0), true);
	EXPECT_EQ(wrapper_array_b->get_element(1), false);
	EXPECT_EQ(wrapper_array_b->get_element(2), true);
}

TEST_F(CDynamicArrayFixture, append_array_int)
{
	wrapper_array_i->reset_array();
	EXPECT_EQ(wrapper_array_i->get_num_elements(), 0);
	wrapper_array_i->append_element(0);
	wrapper_array_i->append_element(1);
	wrapper_array_i->append_element(2);
	EXPECT_EQ(wrapper_array_i->get_num_elements(), 3);
	for (int32_t i = 0; i < 3; i++)
	{
		EXPECT_EQ(wrapper_array_i->get_element(i), i);
	}
}

TEST_F(CDynamicArrayFixture, back_operation)
{
	// test if can achieve the last element
	EXPECT_EQ(wrapper_array_i->back(), 4);
	EXPECT_EQ(wrapper_array_b->back(), true);
}

TEST_F(CDynamicArrayFixture, set_operation)
{
	// test if can set array
	wrapper_array_i->set_element(1, 4);
	wrapper_array_b->set_element(false, 4);
	EXPECT_EQ(wrapper_array_i->get_element(4), 1);
	EXPECT_EQ(wrapper_array_b->get_element(4), false);
}

TEST_F(CDynamicArrayFixture, pop_operation)
{
	// test if can set array
	wrapper_array_i->pop_back();
	wrapper_array_b->pop_back();
	EXPECT_EQ(wrapper_array_i->back(), 3);
	EXPECT_EQ(wrapper_array_b->back(), true);
}

TEST_F(CDynamicArrayFixture, insert_operation)
{
	// test if can set array
	wrapper_array_i->insert_element(8, 2);
	wrapper_array_b->insert_element(false, 2);
	EXPECT_EQ(wrapper_array_i->get_element(2), 8);
	EXPECT_EQ(wrapper_array_b->get_element(2), false);
}

TEST_F(CDynamicArrayFixture, append_array_bool)
{
	wrapper_array_b->reset_array();
	EXPECT_EQ(wrapper_array_b->get_num_elements(), 0);
	wrapper_array_b->append_element(true);
	wrapper_array_b->append_element(false);
	wrapper_array_b->append_element(true);
	EXPECT_EQ(wrapper_array_b->get_num_elements(), 3);
	EXPECT_EQ(wrapper_array_b->get_element(0), true);
	EXPECT_EQ(wrapper_array_b->get_element(1), false);
	EXPECT_EQ(wrapper_array_b->get_element(2), true);
}

TYPED_TEST(CDynamicArrayTest, save_serializable)
{
	for (int32_t i = 0; i < 5; i++)
	{
		this->custom_array->push_back((TypeParam)CMath::random(0, 1));
	}

	/* generate file name */
	char filename[] = "serialization-asciiCDynamicArray.XXXXXX";
	generate_temp_filename(filename);

	CSerializableAsciiFile* file = new CSerializableAsciiFile(filename, 'w');
	this->custom_array->save_serializable(file);
	file->close();
	SG_UNREF(file);

	file = new CSerializableAsciiFile(filename, 'r');
	CDynamicArray<TypeParam>* new_array = new CDynamicArray<TypeParam>();
	new_array->load_serializable(file);
	file->close();
	SG_UNREF(file);

	ASSERT(this->custom_array->get_num_elements() == 5)
	for (int32_t i = 0; i < this->custom_array->get_num_elements(); i++)
	{
		EXPECT_EQ(
		    this->custom_array->get_element(i), new_array->get_element(i));
	}

	SG_UNREF(new_array);
	unlink(filename);
}
