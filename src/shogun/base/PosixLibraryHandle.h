#ifndef POSIX_LIBRAY_HANDLE
#define POSIX_LIBRAY_HANDLE

/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Sanuj Sharma, Viktor Gal
 */

#include <shogun/io/SGIO.h>

#include <string_view>

#include <dlfcn.h>

namespace shogun
{
    namespace internal
    {
        class LibraryHandle
        {
        public:
            LibraryHandle(std::string_view filename): m_filename(filename)
            {
                dlerror();
                m_handle = dlopen(m_filename.data(), RTLD_NOW | RTLD_LOCAL);
                if (m_handle)
                {
                    SG_DEBUG("Loaded {} plugin (at {}).", m_filename, fmt::ptr(m_handle));
                }
                else
                {
                    fmt::memory_buffer msg;
                    fmt::format_to(msg, "Failed loading {}: {}.", m_filename, dlerror());
                    throw std::invalid_argument(fmt::string_view(msg.data(), msg.size()).data());
                }
            }

            LibraryHandle(const LibraryHandle& other):
                m_handle(other.m_handle),
                m_filename(other.m_filename)
            {
            }

            ~LibraryHandle()
            {
                // TODO: we are not closing m_handle here
                // because we dont know whether LibraryHandle
                // is used by any class. In order to close the handle
                // use close() explicitly
            }

            template <typename T>
            T call(std::string_view name)
            {
                if (!m_handle)
                    error("LibraryHandle is invalid as handle is null");

                dlerror();
                T (*fm)();
                *(void**)(&fm) = dlsym(m_handle, name.data());
                char* potential_error = dlerror();
                if (potential_error)
                {
                    fmt::memory_buffer msg;
                    fmt::format_to(msg, "Failed: {}", potential_error);
                    throw std::invalid_argument(fmt::string_view(msg.data(), msg.size()).data());
                }
                return fm();
            }

            /** Equality operator
             * @param first first LibraryHandle
             * @param second second LibraryHandle
             */
            friend bool operator==(const LibraryHandle& first, const LibraryHandle& second)
            {
                return (first.m_handle == second.m_handle)
                    && (first.m_filename == second.m_filename);
            }

            /** Inequality operator
             * @param first first LibraryHandle
             * @param second second LibraryHandle
             */
            friend bool operator!=(const LibraryHandle& first, const LibraryHandle& second)
            {
                return !(first == second);
            }

            /** Explicitly close this handl
             * DANGERZZZZZ!!! LibraryHandle becomes invalid
             * you should destruct this object, NOW!
             */
            void close()
            {
                if (m_handle)
                {
                    SG_DEBUG("Closing {} plugin.", m_filename);
                    dlclose(m_handle);
                }
                m_handle = nullptr;
            }

        private:
            void* m_handle;
            std::string m_filename;
        };
    }
}

#endif
