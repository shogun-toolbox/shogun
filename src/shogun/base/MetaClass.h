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
        spawn_function(recall_type<SpawnFunction>(sf))
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
