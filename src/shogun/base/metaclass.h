/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Sanuj Sharma
 */

#ifndef _META_CLASS_H_
#define _META_CLASS_H_

#include <shogun/lib/any.h>
#include <shogun/base/unique.h>
#include <shogun/base/some.h>

#include <memory>

namespace shogun
{
    /** @brief Metaclass provides an API to
    * spawn shared-pointer-like objects of typename
    * of a Metaclass object.
    */
    template <typename T>
    class MetaClass
    {
        typedef std::function<Some<T>()> SpawnFunction;

    public:
        /** Constructor
        * @param sf Any object of SpawnFunction
        */
        MetaClass(Any sf) :
        spawn_function(any_cast<SpawnFunction>(sf))
        {
        }

        /** Copy Constructor
        * @param other MetaClass object to be copied
        */
        MetaClass(const MetaClass<T>& other) :
        spawn_function(other.spawn_function)
        {
        }

        /** Assignment operator
        * @param other MetaClass object to be assigned
        */
        MetaClass& operator=(const MetaClass<T>& other)
        {
            spawn_function = other.spawn_function;
            return *this;
        }

        /** Destructor */
        ~MetaClass()
        {
        }

        /** Equality operator
        * @param other MetaClass object
        * @return true if both are equal
        */
        bool operator==(const MetaClass<T>& other) const
        {
            return true;
        }

        /** Inequality operator
        * @param other MetaClass object
        * @return false if both are equal
        */
        bool operator!=(const MetaClass<T>& other) const
        {
            return !(*this == other);
        }

        /** @return instance of typename of MetaClass object */
        Some<T> instance() const
        {
            return spawn_function();
        }

    private:
        const SpawnFunction spawn_function;
    };
}

#endif	//_META_CLASS_H_
