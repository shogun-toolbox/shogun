/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Sanuj Sharma
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
