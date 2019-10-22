/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#include "utils/Utils.h"
#include "utils/SGObjectIterator.h"
#include <gtest/gtest.h>
#include <iterator>
#include <shogun/base/ShogunEnv.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/class_list.h>
#include <shogun/base/range.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/serialization/JsonSerializer.h>
#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/stream/FileInputStream.h>
#include <shogun/io/stream/FileOutputStream.h>

using namespace shogun;

// list of classes that (currently) cannot be instantiated
std::set<std::string> sg_object_all_ignores = {"ParseBuffer", "Set",
                                               "TreeMachine"};

// template arguments for SGObject
// TODO: SGString doesn't support complex128_t, so omitted here
typedef ::testing::Types<bool, char, int8_t, int16_t, int32_t, int64_t,
                         float32_t, float64_t, floatmax_t, untemplated_sgobject>
    SGPrimitiveTypes;

template <typename T>
class SGObjectAll : public ::testing::Test
{
};

TYPED_TEST_CASE(SGObjectAll, SGPrimitiveTypes);

TYPED_TEST(SGObjectAll, sg_object_iterator)
{
	for (auto obj : sg_object_iterator<TypeParam>())
	{
		ASSERT_NE(obj, nullptr);
		SCOPED_TRACE(obj->get_name());
		ASSERT_EQ(2, obj.use_count());
	}
}

TYPED_TEST(SGObjectAll, clone_basic)
{
	for (auto obj : sg_object_iterator<TypeParam>().ignore(sg_object_all_ignores))
	{
		SCOPED_TRACE(obj->get_name());
		std::shared_ptr<SGObject> clone;
		try
		{
			clone = obj->clone();
		}
		catch (...)
		{
		}

		ASSERT_NE(clone, nullptr);
		EXPECT_NE(clone, obj);
		EXPECT_EQ(1, clone.use_count());
		EXPECT_EQ(std::string(clone->get_name()), std::string(obj->get_name()));
	}
}

TYPED_TEST(SGObjectAll, clone_equals_empty)
{
	for (auto obj : sg_object_iterator<TypeParam>().ignore(sg_object_all_ignores))
	{
		SCOPED_TRACE(obj->get_name());

		auto clone = obj->clone();
		EXPECT_TRUE(clone->equals(obj));
	}
}

TYPED_TEST(SGObjectAll, serialization_empty_json)
{
	for (auto obj : sg_object_iterator<TypeParam>().ignore(sg_object_all_ignores))
	{
		SCOPED_TRACE(obj->get_name());

		std::string filename = "shogun-unittest-serialization-json-" +
		                       std::string(obj->get_name()) + "_" +
		                       sg_primitive_type_string<TypeParam>() +
		                       ".XXXXXX";

		generate_temp_filename(const_cast<char*>(filename.c_str()));

		auto fs = env();
		ASSERT_FALSE(fs->file_exists(filename));
		std::unique_ptr<io::WritableFile> file;
		ASSERT_FALSE(fs->new_writable_file(filename, &file));
		auto fos = std::make_shared<io::FileOutputStream>(file.get());
		auto serializer = std::make_unique<io::JsonSerializer>();
		serializer->attach(fos);
		serializer->write(obj);

		std::unique_ptr<io::RandomAccessFile> raf;
		ASSERT_FALSE(fs->new_random_access_file(filename, &raf));
		auto fis = std::make_shared<io::FileInputStream>(raf.get());
		auto deserializer = std::make_unique<io::JsonDeserializer>();
		deserializer->attach(fis);
		auto loaded = deserializer->read_object();

		// set accuracy to tolerate lossy formats
		env()->set_global_fequals_epsilon(1e-14);
		ASSERT_TRUE(obj->equals(loaded));
		env()->set_global_fequals_epsilon(0);
		ASSERT_FALSE(fs->delete_file(filename));
	}
}

