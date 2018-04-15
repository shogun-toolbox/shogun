/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Sanuj Sharma
 */

#ifndef _MANIFEST_H_
#define _MANIFEST_H_

#include <shogun/base/metaclass.h>
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
            return any_cast<MetaClass<T>>(clazz);
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
