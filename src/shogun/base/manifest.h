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

#ifndef _MANIFEST_H_
#define _MANIFEST_H_

#include <shogun/base/MetaClass.h>
#include <shogun/base/unique.h>
#include <shogun/base/some.h>

namespace shogun
{

    class Library;

    /** @brief Manifest stores meta-data of Library.
     * Each manifest has description and a set of meta-classes
     * (see @ref MetaClass) which are responsible for
     * creating instances of exported classes.
     */
    class Manifest
    {
    public:
        /** Constructor to initialize hash from name
         * @param description description for the Library
         * @param metaclasses list of meta-classes for exported classes
         */
        Manifest(const std::string& description,
            const std::initializer_list<std::pair<std::string,Any>> metaclasses);
        
        /** Copy constructor
         * @param other Manifest object to be copied
         */
        Manifest(const Manifest& other);

        /** Class Assignment operator
         * @param other manifest object to be assigned
         */
        Manifest& operator=(const Manifest& other);

        /** Equality operator
         * @param first first Manifest
         * @param second second Manifest
         */
        friend bool operator==(const Manifest& first, const Manifest& second);

        /** Inequality operator
         * @param first first Manifest
         * @param second second Manifest
         */
        friend bool operator!=(const Manifest& first, const Manifest& second);
        
        /** Destructor */
        ~Manifest();

        /** Returns meta-class by its name.
         *
         * @param name name of meta-class to obtain
         * @return object of meta-class
         */
        template <typename T>
        MetaClass<T> class_by_name(const std::string& name) const
        {
            Any clazz = find_class(name);
            return recall_type<MetaClass<T>>(clazz);
        }

        /** @return description stored in the manifest. */
        std::string description() const;

    protected:
        /** Adds mapping from class name to MetaClass object of
         * class (stored as Any object) corresponding to the name.
         * The map is stored in Self.
         * @param name name for class
         * @param clazz class
         */
        void add_class(const std::string& name, Any clazz);

        /** Finds MetaClass object (stored as Any object) of class
         * corresponding to the input name.
         * @param name name for class
         * @return 
         */
        Any find_class(const std::string& name) const;
    
    private:
        class Self;
        Unique<Self> self;
    };

/** Starts manifest declaration with its description.
 * Always immediately follow this macro with
 * @ref EXPORT or @ref END_MANIFEST.
 */
#define BEGIN_MANIFEST(DESCRIPTION)                             \
extern "C" Manifest shogunManifest()                            \
{                                                               \
    static Manifest manifest(DESCRIPTION,{                      \

/** Declares class to be exported.
 * Always use this macro between @ref BEGIN_MANIFEST and
 * @ref END_MANIFEST
 */
#define EXPORT(CLASSNAME, BASE_CLASSNAME, IDENTIFIER)           \
    std::make_pair(IDENTIFIER, erase_type(                      \
        MetaClass<BASE_CLASSNAME>(erase_type(                   \
            std::function<Some<BASE_CLASSNAME>()>(              \
                []() -> Some<BASE_CLASSNAME>                    \
                {                                               \
                    return Some<BASE_CLASSNAME>(new CLASSNAME); \
                }                                               \
                ))))),                                          \

/** Ends manifest declaration.
 * Always use this macro after @ref BEGIN_MANIFEST
 */
#define END_MANIFEST()                                          \
        });                                                     \
    return manifest;                                            \
}

}

#endif //_MANIFEST_H_
