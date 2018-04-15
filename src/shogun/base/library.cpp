/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Sanuj Sharma
 */

#include <shogun/base/library.h>
#include <shogun/lib/common.h>

namespace shogun
{
    class LibraryHandle : public SGObject
    {
    public:
        LibraryHandle(const std::string& filename)
        {
            dlerror();
            handle = dlopen(filename.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (handle)
            {
                io::info("Loaded.");
            }
            else
            {
                error("Failed: {}.", dlerror());
            }
        }

        ~LibraryHandle()
        {
            if (handle)
            {
                io::info("Closing library.");
                dlclose(handle);
            }
        }

        template <typename T>
        T call(const std::string& name)
        {
            dlerror();
            T (*fm)();
            *(void**)(&fm) = dlsym(handle, name.c_str());
            char* potential_error = dlerror();
            if (potential_error)
                error("Failed: {}.", potential_error);
            return fm();
        }

        const char* get_name() const
        {
            return "LibraryHandle";
        }

    private:
        void* handle;

    };

    Library::Library(const std::string& filename) :
        m_handle(std::make_shared<LibraryHandle>(filename))
    {
    }

    Library::Library(const Library& other) :
        m_handle(std::make_shared<LibraryHandle>(other.m_handle))
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
        return first_handle == second_handle;
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
        return m_handle->call<Manifest>(manifest_accessor_name);
    }

    Library load_library(const std::string& filename)
    {
        return Library(filename);
    }

    const char* Library::manifest_accessor_name = "shogunManifest";

}
