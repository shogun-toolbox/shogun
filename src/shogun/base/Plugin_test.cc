/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2016 Sanuj Sharma
 */
#include <gtest/gtest.h>

#include "shogun/util/MockBaseClass.h"
#include <shogun/kernel/GaussianKernel.h>

#include <sstream>

using namespace shogun;
using namespace std;

static string library_name(string_view library_name)
{
	ostringstream ss;
	ss << SHARED_LIB_PREFIX << library_name << SHARED_LIB_SUFFIX;
	return ss.str();
}

TEST(Plugin, load_close)
{
	Library library = load_library(library_name("plugin"));
	{
		auto manifest = library.manifest();
		auto mock_class = manifest.class_by_name<MockBaseClass>("mock_class");
		auto another_mock_class = manifest.class_by_name<MockBaseClass>("another_mock_class");
		auto mock_class_obj = mock_class.instance();
		auto another_mock_class_obj = another_mock_class.instance();

		std::string mock_class_name = mock_class_obj->get_name();
		std::string another_mock_class_name = another_mock_class_obj->get_name();

		EXPECT_EQ(0, mock_class_obj->mock_method());
		EXPECT_EQ(0, another_mock_class_obj->mock_method());
		EXPECT_EQ("MockClass", mock_class_name);
		EXPECT_EQ("AnotherMockClass", another_mock_class_name);
	}
	unload_library(std::move(library));
}

TEST(Plugin, load_non_existent_lib)
{
	EXPECT_THROW(load_library(library_name("randomplugin")), ShogunException);
}

TEST(Plugin, load_non_existent_class)
{
	auto library = load_library(library_name("plugin"));
	{
		auto manifest = library.manifest();
		EXPECT_THROW(manifest.class_by_name<MockBaseClass>("arbitrary_class"), ShogunException);
	}
	unload_library(std::move(library));
}

TEST(Plugin, manifest_equality_operator)
{
	Manifest manifest1("Mock library",
	{
		make_pair("mock_base_class", make_any(
			MetaClass<SGObject>(make_any<shared_ptr<SGObject>>(
					[]()
					{
						return shared_ptr<SGObject>(new MockBaseClass);
					}
					)))),
	});

	Manifest manifest2("Mock library",
	{
		make_pair("mock_base_class", make_any(
			MetaClass<SGObject>(make_any<shared_ptr<SGObject>>(
					[]()
					{
						return shared_ptr<SGObject>(new MockBaseClass);
					}
					)))),
	});

	Manifest manifest3("Library",
	{
		make_pair("mock_base_class", make_any(
			MetaClass<SGObject>(make_any<shared_ptr<SGObject>>(
					[]()
					{
						return shared_ptr<SGObject>(new MockBaseClass);
					}
					)))),
	});

	EXPECT_EQ("Mock library", manifest1.description());
	EXPECT_EQ(manifest1, manifest2);
	EXPECT_NE(manifest1, manifest3);
}

TEST(Plugin, manifest_copy_constructor)
{
	Manifest manifest("Mock library",
	{
		make_pair("mock_base_class", make_any(
			MetaClass<SGObject>(make_any<shared_ptr<SGObject>>(
					[]()
					{
						return shared_ptr<SGObject>(new MockBaseClass);
					}
					)))),
	});

	Manifest copy_manifest(manifest);

	EXPECT_EQ(manifest, copy_manifest);
}

TEST(Plugin, manifest_assignment_operator)
{
	Manifest manifest("Mock library",
	{
		make_pair("mock_base_class", make_any(
			MetaClass<SGObject>(make_any<shared_ptr<SGObject>>(
					[]()
					{
						return shared_ptr<SGObject>(new MockBaseClass);
					}
					)))),
	});

	Manifest assigned_manifest = manifest;

	EXPECT_EQ(manifest, assigned_manifest);
}

TEST(Plugin, load_lib_wo_manifest)
{
	auto library = load_library(library_name("pluginwomanifest"));
	{
		EXPECT_THROW(library.manifest(), ShogunException);
	}
	unload_library(std::move(library));
}

TEST(Plugin, library_copy_constructor)
{
	Library library(library_name("plugin"));
	{
		Library copy_library(library);
		EXPECT_EQ(library, copy_library);
	}
	unload_library(std::move(library));
}

TEST(Plugin, library_assignment_operator)
{
	Library library(library_name("plugin"));
	{
		Library assigned_library = library;
		EXPECT_EQ(library, assigned_library);
	}
	unload_library(std::move(library));
}

TEST(Plugin, library_manifest_accessor_name)
{
	auto name = Library::get_manifest_accessor_name();
	EXPECT_EQ("shogunManifest", name);
}

TEST(Plugin, metaclass_equality_operator)
{
	MetaClass<SGObject> meta_class(make_any<shared_ptr<SGObject>>(
			[]()
			{
				return shared_ptr<SGObject>(new MockBaseClass);
			}
		));

	MetaClass<SGObject> another_meta_class(make_any<shared_ptr<SGObject>>(
			[]()
			{
				return shared_ptr<SGObject>(new MockBaseClass);
			}
		));

	EXPECT_EQ(meta_class, another_meta_class);
}

TEST(Plugin, metaclass_copy_constructor)
{
	MetaClass<SGObject> meta_class(make_any<shared_ptr<SGObject>>(
			[]()
			{
				return shared_ptr<SGObject>(new MockBaseClass);
			}
		));

	MetaClass<SGObject> copy_meta_class(meta_class);

	EXPECT_EQ(meta_class, copy_meta_class);
}

TEST(Plugin, metaclass_assignment_operator)
{
	MetaClass<SGObject> meta_class(make_any<shared_ptr<SGObject>>(
			[]()
			{
				return shared_ptr<SGObject>(new MockBaseClass);
			}
		));

	MetaClass<SGObject> assigned_meta_class = meta_class;

	EXPECT_EQ(meta_class, assigned_meta_class);
}