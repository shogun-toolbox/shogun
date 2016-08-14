/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2016 Sanuj Sharma
 */

#include <../tests/unit/base/MockBaseClass.h>
#include <shogun/kernel/GaussianKernel.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(Plugin, load_close)
{
    Library library = load_library("./libplugin.so");
    Manifest manifest = library.manifest();
    MetaClass<MockBaseClass> mock_class = manifest.class_by_name<MockBaseClass>("mock_class");
    MetaClass<MockBaseClass> another_mock_class = manifest.class_by_name<MockBaseClass>("another_mock_class");
    Some<MockBaseClass> mock_class_obj = mock_class.instance();
    Some<MockBaseClass> another_mock_class_obj = another_mock_class.instance();

    std::string mock_class_name = mock_class_obj->get_name();
    std::string another_mock_class_name = another_mock_class_obj->get_name();

    EXPECT_EQ(mock_class_obj->mock_method(), 0);
    EXPECT_EQ(another_mock_class_obj->mock_method(), 0);
    EXPECT_EQ(mock_class_name, "MockClass");
    EXPECT_EQ(another_mock_class_name, "AnotherMockClass");
}

TEST(Plugin, load_non_existent_lib)
{
    EXPECT_THROW(load_library("./librandomplugin.so"), ShogunException);
}

TEST(Plugin, load_non_existent_class)
{
    auto library = load_library("./libplugin.so");
    auto manifest = library.manifest();
    EXPECT_THROW(manifest.class_by_name<MockBaseClass>("arbitrary_class"), ShogunException);
}

TEST(Plugin, manifest_equality_operator)
{
    Manifest manifest1("Mock library",
    {
        std::make_pair("mock_base_class", erase_type(
            MetaClass<CSGObject>(erase_type(
                std::function<Some<CSGObject>()>(
                    []() -> Some<CSGObject>
                    {
                        return Some<CSGObject>(new MockBaseClass);
                    }
                    ))))),
    });

    Manifest manifest2("Mock library",
    {
        std::make_pair("mock_base_class", erase_type(
            MetaClass<CSGObject>(erase_type(
                std::function<Some<CSGObject>()>(
                    []() -> Some<CSGObject>
                    {
                        return Some<CSGObject>(new MockBaseClass);
                    }
                    ))))),
    });

    Manifest manifest3("Library",
    {
        std::make_pair("mock_base_class", erase_type(
            MetaClass<CSGObject>(erase_type(
                std::function<Some<CSGObject>()>(
                    []() -> Some<CSGObject>
                    {
                        return Some<CSGObject>(new MockBaseClass);
                    }
                    ))))),
    });

    EXPECT_EQ(manifest1.description(), "Mock library");
    EXPECT_EQ(manifest2, manifest1);
    EXPECT_NE(manifest3, manifest1);
}

TEST(Plugin, manifest_copy_constructor)
{
    Manifest manifest("Mock library",
    {
        std::make_pair("mock_base_class", erase_type(
            MetaClass<CSGObject>(erase_type(
                std::function<Some<CSGObject>()>(
                    []() -> Some<CSGObject>
                    {
                        return Some<CSGObject>(new MockBaseClass);
                    }
                    ))))),
    });

    Manifest copy_manifest(manifest);

    EXPECT_EQ(copy_manifest, manifest);
}

TEST(Plugin, manifest_assignment_operator)
{
    Manifest manifest("Mock library",
    {
        std::make_pair("mock_base_class", erase_type(
            MetaClass<CSGObject>(erase_type(
                std::function<Some<CSGObject>()>(
                    []() -> Some<CSGObject>
                    {
                        return Some<CSGObject>(new MockBaseClass);
                    }
                    ))))),
    });

    Manifest assigned_manifest = manifest;

    EXPECT_EQ(assigned_manifest, manifest);
}

TEST(Plugin, load_lib_wo_manifest)
{
    auto library = load_library("./libpluginwomanifest.so");
    EXPECT_THROW(library.manifest(), ShogunException);
}

TEST(Plugin, library_copy_constructor)
{
    Library library("./libplugin.so");
    Library copy_library(library);
    EXPECT_EQ(copy_library, library);
}

TEST(Plugin, library_assignment_operator)
{
    Library library("./libplugin.so");
    Library assigned_library = library;
    EXPECT_EQ(assigned_library, library);
}

TEST(Plugin, library_manifest_accessor_name)
{
    std::string name = Library::get_manifest_accessor_name();
    EXPECT_EQ(name, "shogunManifest");
}
    
TEST(Plugin, metaclass_equality_operator)
{
    MetaClass<CSGObject> meta_class(erase_type(
        std::function<Some<CSGObject>()>(
            []() -> Some<CSGObject>
            {
                return Some<CSGObject>(new MockBaseClass);
            }
        )));

    MetaClass<CSGObject> another_meta_class(erase_type(
        std::function<Some<CSGObject>()>(
            []() -> Some<CSGObject>
            {
                return Some<CSGObject>(new MockBaseClass);
            }
        )));

    EXPECT_EQ(another_meta_class, meta_class);
}

TEST(Plugin, metaclass_copy_constructor)
{
    MetaClass<CSGObject> meta_class(erase_type(
        std::function<Some<CSGObject>()>(
            []() -> Some<CSGObject>
            {
                return Some<CSGObject>(new MockBaseClass);
            }
        )));

    MetaClass<CSGObject> copy_meta_class(meta_class);

    EXPECT_EQ(copy_meta_class, meta_class);
}

TEST(Plugin, metaclass_assignment_operator)
{
    MetaClass<CSGObject> meta_class(erase_type(
        std::function<Some<CSGObject>()>(
            []() -> Some<CSGObject>
            {
                return Some<CSGObject>(new MockBaseClass);
            }
        )));

    MetaClass<CSGObject> assigned_meta_class = meta_class;

    EXPECT_EQ(assigned_meta_class, meta_class);
}
