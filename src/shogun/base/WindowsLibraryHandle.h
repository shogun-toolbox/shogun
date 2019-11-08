/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#ifndef WINDOWS_LIBRAY_HANDLE
#define WINDOWS_LIBRAY_HANDLE

#include <shogun/io/SGIO.h>

#include <string>
#include <string_view>

#include <Shlwapi.h>
#undef StrCat  // Don't let StrCat be renamed to lstrcatA
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <time.h>
#include <windows.h>

namespace shogun
{
    namespace internal
    {
        class LibraryHandle
        {
        public:
            LibraryHandle(std::string_view filename): m_filename(filename)
            {
                std::replace(m_filename.begin(), m_filename.end(), '/', '\\');
                std::wstring ws_file_name(utf8_to_wchar(m_filename));

                HMODULE hModule =
                    LoadLibraryExW(ws_file_name.c_str(), NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
                if (hModule)
                {
                    io::info("Loaded {} plugin.", m_filename);
                    m_handle = hModule;
                }
                else
                {
                    fmt::memory_buffer msg;
                    fmt::format_to(msg, "Failed loading {}: {}.", m_filename, GetLastError());
                    throw std::invalid_argument(fmt::string_view(msg.data(), msg.size()).data());
                }
            }

            ~LibraryHandle()
            {
            }

            template <typename T>
            T call(std::string_view name)
            {
                T (*fm)();

                FARPROC found_symbol = GetProcAddress((HMODULE)m_handle, name.data());
                if (found_symbol == nullptr)
                {
                    fmt::memory_buffer msg;
                    fmt::format_to(msg, "Failed finding {} symbol: {}",
                        name, GetLastError());
                    throw std::invalid_argument(fmt::string_view(msg.data(), msg.size()).data());
                }

                *(void**)(&fm) = (void**)found_symbol;
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
                    FreeLibrary(m_handle);
                }
                m_handle = nullptr;
            }

        private:
            void* m_handle;
            std::string_view m_filename;
        };
    }
}

#endif
