/*
 * Copyright (c) 2016, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2016 Sergey Lisitsyn
 * Written (W) 2016 Sanuj Sharma
 */

#ifndef _LIBRARY_H_
#define _LIBRARY_H_

#include <shogun/base/manifest.h>
#include <shogun/base/SGObject.h>
#include <string>
#include <dlfcn.h>

namespace shogun
{

    /** @brief
     * Handles loading, calling and closing of plugins from shared object files.
     */
    class LibraryHandle;

    /** @brief
     * Provides an API for loading plugins as objects of this class
     * and accessing Manifest of the loaded plugins.
     * Uses LibraryHandle under the hood.
     */
    class Library
    {
    public:
        /** Constructor to initialize library
         * @param filename name of shared object file
         */
        Library(const std::string& filename);

        /** Copy constructor
         * @param other library object to be copied
         */
        Library(const Library& other);

        /** Class Assignment operator
         * @param other library object to be assigned
         */
        Library& operator=(const Library& other);

        /** Equality operator
         * @param first first Library
         * @param second second Library
         */
        friend bool operator==(const Library& first, const Library& second);

        /** Inequality operator
         * @param first first Library
         * @param second second Library
         */
        friend bool operator!=(const Library& first, const Library& second);

        /** Destructor */
        ~Library();

        /** @return manifest of loaded library */
        Manifest manifest();

        /** @return name of function that accesses Manifest
         * of loaded library.
         */
        static const char* get_manifest_accessor_name()
        {
            return manifest_accessor_name;
        }

    private:
        static const char* manifest_accessor_name;
        Some<LibraryHandle> m_handle;

    };

    /** Loads a plugin into a library object.
     * @param filename name of shared object file
     * @return library object of loaded plugin
     */
    Library load_library(const std::string& filename);

}

#endif //_LIBRARY_H_
