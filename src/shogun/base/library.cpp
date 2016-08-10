#include <shogun/base/library.h>
#include <shogun/lib/common.h>

namespace shogun
{
    class LibraryHandle : public CSGObject
    {
    
    public:
        LibraryHandle(const std::string& filename)
        {
            dlerror();
            handle = dlopen(filename.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (handle) 
            {
                SG_INFO("Loaded.\n");
            }
            else
            {
                SG_ERROR("Failed: %s.\n", dlerror());
            }
        }

        ~LibraryHandle()
        {
            if (handle)
            {
                SG_INFO("Closing library.\n");
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
                SG_ERROR("Failed: %s.\n", potential_error);
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
        m_handle(some<LibraryHandle>(filename))
    {
    }

    Library::Library(const Library& other) : 
        m_handle(Some<LibraryHandle>(other.m_handle))
    {
    }

    Library& Library::operator=(const Library& other)
    {
        m_handle = other.m_handle;
        return *this;
    }

    bool operator==(const Library& first, const Library& second)
    {
        Some<LibraryHandle> first_handle = first.m_handle;
        Some<LibraryHandle> second_handle = second.m_handle;
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
