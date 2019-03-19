/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#include "utils/Utils.h"
#include "utils/SGObjectIterator.h"
#include <gtest/gtest.h>
#include <iterator>
#include <shogun/base/Parameter.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/class_list.h>
#include <shogun/base/range.h>
#include <shogun/base/some.h>
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
		ASSERT_EQ(obj->ref_count(), 1);
	}
}

TYPED_TEST(SGObjectAll, clone_basic)
{
	for (auto obj : sg_object_iterator<TypeParam>().ignore(sg_object_all_ignores))
	{
		SCOPED_TRACE(obj->get_name());
		CSGObject* clone = nullptr;
		try
		{
			clone = obj->clone();
		}
		catch (...)
		{
		}

		ASSERT_NE(clone, nullptr);
		EXPECT_NE(clone, obj);
		EXPECT_EQ(clone->ref_count(), 1);
		EXPECT_EQ(std::string(clone->get_name()), std::string(obj->get_name()));

		SG_UNREF(clone);
	}
}

TYPED_TEST(SGObjectAll, clone_equals_empty)
{
	for (auto obj : sg_object_iterator<TypeParam>().ignore(sg_object_all_ignores))
	{
		SCOPED_TRACE(obj->get_name());

		CSGObject* clone = obj->clone();
		EXPECT_TRUE(clone->equals(obj));

		SG_UNREF(clone);
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

		SG_REF(obj);
		auto fs = io::FileSystemRegistry::instance();
		ASSERT_FALSE(fs->file_exists(filename));
		std::unique_ptr<io::WritableFile> file;
		ASSERT_FALSE(fs->new_writable_file(filename, &file));
		auto fos = some<io::CFileOutputStream>(file.get());
		auto serializer = some<io::CJsonSerializer>();
		serializer->attach(fos);
		serializer->write(wrap<CSGObject>(obj));

		std::unique_ptr<io::RandomAccessFile> raf;
		ASSERT_FALSE(fs->new_random_access_file(filename, &raf));
		auto fis = some<io::CFileInputStream>(raf.get());
		auto deserializer = some<io::CJsonDeserializer>();
		deserializer->attach(fis);
		auto loaded = deserializer->read();

		// set accuracy to tolerate lossy formats
		set_global_fequals_epsilon(1e-14);
		ASSERT_TRUE(obj->equals(loaded));
		set_global_fequals_epsilon(0);
		ASSERT_FALSE(fs->delete_file(filename));
	}
}

// temporary test until old parameter framework is gone
// enable test to hunt for parameters not registered in tags
// see https://github.com/shogun-toolbox/shogun/issues/4117
// not typed as all template instantiations will have the same tags
TEST(SGObjectAll, DISABLED_tag_coverage)
{
	auto class_names = available_objects();

	for (auto class_name : class_names)
	{
		auto obj = create(class_name.c_str(), PT_NOT_GENERIC);

		// templated classes cannot be created in the above way
		if (!obj)
		{
			// only test single generic type here: all types have the same
			// parameter names
			obj = create(class_name.c_str(), PT_FLOAT64);
		}

		// obj must exist now, whether templated or not
		ASSERT_NE(obj, nullptr);

		// old parameter framework names
		std::vector<std::string> old_names;
		for (auto i : range(obj->m_parameters->get_num_parameters()))
			old_names.push_back(obj->m_parameters->get_parameter(i)->m_name);

		std::vector<std::string> tag_names;
		std::transform(obj->get_params().cbegin(), obj->get_params().cend(), std::back_inserter(tag_names),
			[](const std::pair<std::string, std::shared_ptr<const AnyParameter>>& each) -> std::string {
			return each.first;
		});

		// hack to increase readability of error messages
		old_names.push_back("_Shogun class: " + class_name);
		tag_names.push_back("_Shogun class: " + class_name);

		// comparing std::vector depends on order
		std::sort(old_names.begin(), old_names.end());
		std::sort(tag_names.begin(), tag_names.end());

		EXPECT_EQ(tag_names, old_names);

		SG_UNREF(obj);
	}
}
