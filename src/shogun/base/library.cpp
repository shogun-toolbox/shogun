/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Sanuj Sharma
 */

#include <shogun/base/library.h>
#include <shogun/lib/common.h>

#ifdef _WIN32
#include <shogun/base/WindowsLibraryHandle.h>
#else
#include <shogun/base/PosixLibraryHandle.h>
#endif

namespace shogun
{
    Library::Library(std::string_view filename) :
        m_handle(std::make_shared<internal::LibraryHandle>(filename))
    {
    }

    Library::Library(const Library& other) :
        m_handle(other.m_handle)
    {
    }

    Library& Library::operator=(const Library& other)
    {
        m_handle = other.m_handle;
        return *this;
    }

    bool operator==(const Library& first, const Library& second)
    {
        auto first_handle = first.m_handle;
        auto second_handle = second.m_handle;
        return *first_handle == *second_handle;
    }

    bool operator!=(const Library& first, const Library& second)
    {
        return !(first == second);
    }

    Library::~Library()
    {
    }

    Manifest Library::manifest()
    {
        return m_handle->call<Manifest>(kManifestAccessorName);
    }

    void Library::close()
    {
        m_handle->close();
    }

    Library load_library(std::string_view filename)
    {
        return Library(filename);
    }

    void unload_library(Library&& lib)
    {
        lib.close();
    }

}
