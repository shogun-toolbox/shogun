/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#include "utils/Utils.h"
#include <gtest/gtest.h>
#include <iterator>
#include <shogun/base/Parameter.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/class_list.h>
#include <shogun/base/range.h>
#include <shogun/base/some.h>
#include <shogun/io/SerializableAsciiFile.h>

using namespace shogun;

// to have a type for non-template SGObject classes
struct untemplated_sgobject
{
};

/** Returns primitive type value for template parameter */
template <class T>
EPrimitiveType sg_primitive_type();
template <class T>
std::string sg_primitive_type_string();

// stringizing the result of expansion of a macro argument used below
// https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html
#define STRING__(s) #s
#define XSTRING__(s) STRING__(s)

#define SG_PRIMITIVE_TYPE(T, ptype)                                            \
	template <>                                                                \
	EPrimitiveType sg_primitive_type<T>()                                      \
	{                                                                          \
		return ptype;                                                          \
	}                                                                          \
                                                                               \
	template <>                                                                \
	std::string sg_primitive_type_string<T>()                                  \
	{                                                                          \
		return std::string(XSTRING__(ptype));                                  \
	}

SG_PRIMITIVE_TYPE(bool, PT_BOOL);
SG_PRIMITIVE_TYPE(char, PT_CHAR);
SG_PRIMITIVE_TYPE(int8_t, PT_INT8);
SG_PRIMITIVE_TYPE(uint8_t, PT_UINT8);
SG_PRIMITIVE_TYPE(int16_t, PT_INT16);
SG_PRIMITIVE_TYPE(int32_t, PT_INT32);
SG_PRIMITIVE_TYPE(int64_t, PT_INT64);
SG_PRIMITIVE_TYPE(float32_t, PT_FLOAT32);
SG_PRIMITIVE_TYPE(float64_t, PT_FLOAT64);
SG_PRIMITIVE_TYPE(floatmax_t, PT_FLOATMAX);
SG_PRIMITIVE_TYPE(untemplated_sgobject, PT_NOT_GENERIC);
#undef SG_PRIMITIVE_TYPE
#undef XSTRING__
#undef STRING__

/** Helper to write c++11 style loops in tests that cover all Shogun classes */
template <typename T>
class sg_object_iterator
{
public:
	sg_object_iterator()
	{
		m_class_names = available_objects();
	}

	sg_object_iterator(std::set<std::string> ignores)
	{
		m_class_names = available_objects();

		std::set<std::string> diff;

		std::set_difference(
		    m_class_names.begin(), m_class_names.end(), ignores.begin(),
		    ignores.end(), std::inserter(diff, diff.begin()));
		m_class_names = diff;
	}

	class Iterator : public std::iterator<std::input_iterator_tag, T>
	{
	public:
		Iterator(
		    std::set<std::string>::iterator it,
		    std::set<std::string>::iterator end)
		    : m_it(it), m_end(end), m_obj(nullptr)
		{
			create_and_forward_if_necessary();
		}
		Iterator(const Iterator& other)
		    : m_it(other.m_it), m_end(other.m_end), m_obj(other.m_obj)
		{
			create_and_forward_if_necessary();
		}
		Iterator(Iterator&& other)
		    : m_it(other.m_it), m_end(other.m_end), m_obj(other.m_obj)
		{
			create_and_forward_if_necessary();
		}
		~Iterator()
		{
			SG_UNREF(m_obj);
		}

		Iterator& operator=(const Iterator&) = delete;
		Iterator& operator++()
		{
			SG_UNREF(m_obj);
			do
			{
				++m_it;

				if (m_it == m_end)
					return *this;

				m_obj = create_obj();
			} while (!m_obj);

			return *this;
		}
		CSGObject* operator*()
		{
			return m_obj;
		}
		bool operator!=(const Iterator& other) const
		{
			return this->m_it != other.m_it;
		}

	private:
		CSGObject* create_obj()
		{
			auto obj = create((*m_it).c_str(), sg_primitive_type<T>());
			SG_REF(obj);
			return obj;
		}

		void create_and_forward_if_necessary()
		{
			if (m_it == m_end)
				return;

			m_obj = create_obj();
			while (!m_obj && m_it != m_end)
			{
				this->operator++();
			}
		}

		std::set<std::string>::iterator m_it;
		std::set<std::string>::iterator m_end;
		CSGObject* m_obj;
	};

	Iterator begin() const
	{
		return Iterator(m_class_names.begin(), m_class_names.end());
	}
	Iterator end() const
	{
		return Iterator(m_class_names.end(), m_class_names.end());
	}

	std::set<std::string> m_class_names;
};

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
	for (auto obj : sg_object_iterator<TypeParam>(sg_object_all_ignores))
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
	for (auto obj : sg_object_iterator<TypeParam>(sg_object_all_ignores))
	{
		SCOPED_TRACE(obj->get_name());

		CSGObject* clone = obj->clone();
		EXPECT_TRUE(clone->equals(obj));

		SG_UNREF(clone);
	}
}

TYPED_TEST(SGObjectAll, serialization_empty_ascii)
{
	for (auto obj : sg_object_iterator<TypeParam>(sg_object_all_ignores))
	{
		SCOPED_TRACE(obj->get_name());

		std::string filename = "shogun-unittest-serialization-ascii-" +
		                       std::string(obj->get_name()) + "_" +
		                       sg_primitive_type_string<TypeParam>() +
		                       ".XXXXXX";

		generate_temp_filename(const_cast<char*>(filename.c_str()));

		auto file_save = some<CSerializableAsciiFile>(filename.c_str(), 'w');
		ASSERT_TRUE(obj->save_serializable(file_save));
		file_save->close();

		CSGObject* loaded = create(obj->get_name(), obj->get_generic());
		ASSERT_NE(loaded, nullptr);
		auto file_load = some<CSerializableAsciiFile>(filename.c_str(), 'r');
		ASSERT_TRUE(loaded->load_serializable(file_load));
		file_load->close();

		// set accuracy to tolerate lossy formats
		set_global_fequals_epsilon(1e-14);
		ASSERT_TRUE(obj->equals(loaded));
		set_global_fequals_epsilon(0);

		SG_UNREF(loaded);

		ASSERT_EQ(unlink(filename.c_str()), 0);
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
		std::transform(obj->get_parameters().cbegin(), obj->get_parameters().cend(), std::back_inserter(tag_names),
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
