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

#ifndef _ANY_H_
#define _ANY_H_

#include <string.h>
#include <stdexcept>
#include <typeinfo>
#ifdef HAVE_CXA_DEMANGLE
#include <cxxabi.h>
#endif

namespace shogun
{
    /** Converts compiler-dependent name of class to
     * something human readable.
     * @return human readable name of class
     */
    template <typename T>
    std::string demangledType()
    {
#ifdef HAVE_CXA_DEMANGLE
		size_t length;
		int status;
		char* demangled = abi::__cxa_demangle(typeid(T).name(), nullptr, &length, &status);
		std::string demangled_string(demangled);
		free(demangled);
#else
		std::string demangled_string(typeid(T).name());
#endif
        return demangled_string;
    }

    enum class PolicyType {
        OWNING,
        NON_OWNING
    };

    /** @brief An interface for a policy to store a value.
     * Value can be any data like primitive data-types, shogun objects, etc.
     * Policy defines how to handle this data. It works with a
     * provided memory region and is able to set value, clear it
     * and return the type-name as string.
     */
    class BaseAnyPolicy
    {
    public:
        /** Puts provided value pointed by v (untyped to be generic) to storage.
         * @param storage pointer to a pointer to storage
         * @param v pointer to value
         */
        virtual void set(void** storage, const void* v) const = 0;

        /** Clears storage.
         * @param storage pointer to a pointer to storage
         */
        virtual void clear(void** storage) const = 0;

        /** Returns type-name as string.
         * @return name of type class
         */
        virtual std::string type() const = 0;

        /** Compares type.
         * @param ti type information
         * @return true if type matches
         */
        virtual bool matches(const std::type_info& ti) const = 0;

        /** Compares two storages.
         * @param storage pointer to a pointer to storage
         * @param other_storage pointer to a pointer to another storage
         * @return true if both storages have same value
         */
        virtual bool equals(void** storage, void** other_storage) const = 0;

        /** Returns the name of policy.
         * @return name of policy
         */
        virtual std::string policy_name() const = 0;

        /** Returns the type of policy.
         * @return type of policy
         */
        virtual PolicyType policy_type() const = 0;
    };

    /** @brief This is one concrete implementation of policy that
     * uses void pointers to store values.
     */
    template <typename T>
    class PointerValueAnyPolicy : public BaseAnyPolicy
    {
    public:
        /** Puts provided value pointed by v (untyped to be generic) to storage.
         * @param storage pointer to a pointer to storage
         * @param v pointer to value
         */
        virtual void set(void** storage, const void* v) const
        {
            *(storage) = new T(*reinterpret_cast<T const*>(v));
        }

        /** Clears storage.
         * @param storage pointer to a pointer to storage
         */
        virtual void clear(void** storage) const
        {
            delete reinterpret_cast<T*>(*storage);
        }

        /** Returns type-name as string.
         * @return name of type class
         */
        virtual std::string type() const
        {
            return demangledType<T>();
        }

        /** Compares type.
         * @param ti type information
         * @return true if type matches
         */
        virtual bool matches(const std::type_info& ti) const
        {
            return typeid(T) == ti;
        }

        /** Compares two storages.
         * @param storage pointer to a pointer to storage
         * @param other_storage pointer to a pointer to another storage
         * @return true if both storages have same value
         */
        bool equals(void** storage, void** other_storage) const
        {
            T typed_storage = *(reinterpret_cast<T*>(*storage));
            T typed_other_storage = *(reinterpret_cast<T*>(*other_storage));
            return typed_storage == typed_other_storage;
        }

        virtual std::string policy_name() const {
            return "owning";
        }

        virtual PolicyType policy_type() const {
            return PolicyType::OWNING;
        }
    };

    template <typename T>
    class NonOwningAnyPolicy : public BaseAnyPolicy
    {
    public:
        /** Puts provided value pointed by v (untyped to be generic) to storage.
         * @param storage pointer to a pointer to storage
         * @param v pointer to value
         */
        virtual void set(void** storage, const void* v) const
        {
            *(storage) = const_cast<void*>(v);
        }

        /** Clears storage.
         * @param storage pointer to a pointer to storage
         */
        virtual void clear(void** storage) const
        {
        }

        /** Returns type-name as string.
         * @return name of type class
         */
        virtual std::string type() const
        {
            return demangledType<T>();
        }

        /** Compares type.
         * @param ti type information
         * @return true if type matches
         */
        virtual bool matches(const std::type_info& ti) const
        {
            return typeid(T) == ti;
        }

        /** Compares two storages.
         * @param storage pointer to a pointer to storage
         * @param other_storage pointer to a pointer to another storage
         * @return true if both storages have same value
         */
        bool equals(void** storage, void** other_storage) const
        {
            T typed_storage = *(reinterpret_cast<T*>(*storage));
            T typed_other_storage = *(reinterpret_cast<T*>(*other_storage));
            return typed_storage == typed_other_storage;
        }

        virtual std::string policy_name() const {
            return "non owning";
        }

        virtual PolicyType policy_type() const {
            return PolicyType::NON_OWNING;
        }
    };

    template <typename T>
    static BaseAnyPolicy* owning_policy()
    {
        typedef PointerValueAnyPolicy<T> Policy;
        static Policy policy;
        return &policy;
    }

    template <typename T>
    static BaseAnyPolicy* non_owning_policy()
    {
        typedef NonOwningAnyPolicy<T> Policy;
        static Policy policy;
        return &policy;
    }

    /** @brief Allows to store objects of arbitrary types
     * by using a BaseAnyPolicy and provides a type agnostic API.
     * See its usage in CSGObject::Self, CSGObject::set(), CSGObject::get()
     * and CSGObject::has().
     * .
     */
    class Any
    {
    public:
        /** Used to denote an empty Any object */
        struct Empty;

        /** Empty value constructor */
        Any() : Any(owning_policy<Empty>(), nullptr)
        {
        }

        /** Constructor to copy value */
        template <typename T>
        explicit Any(const T& v) : Any(owning_policy<T>(), nullptr)
        {
            policy->set(&storage, &v);
        }

        /** Base constructor */
        Any(BaseAnyPolicy* the_policy, void* the_storage) : policy(the_policy), storage(the_storage)
        {
        }

        /** Copy constructor */
        Any(const Any& other) : Any(other.policy, nullptr)
        {
            assert_same_policy_type(other.policy);
            policy->set(&storage, other.storage);
        }

        /** Assignment operator
         * @param other another Any object
         * @return Any object
         */
        Any& operator=(const Any& other)
        {
            assert_same_policy_type(other.policy);
            policy->clear(&storage);
            policy = other.policy;
            policy->set(&storage, other.storage);
            return *(this);
        }

        /** Equality operator
         * @param lhs Any object on left hand side
         * @param rhs Any object on right hand side
         * @return true if both are equal
         */
        friend inline bool operator==(const Any& lhs, const Any& rhs);

        /** Inequality operator
         * @param lhs Any object on left hand side
         * @param rhs Any object on right hand side
         * @return false if both are equal
         */
        friend inline bool operator!=(const Any& lhs, const Any& rhs);

        /** Destructor */
        ~Any()
        {
            policy->clear(&storage);
        }

        /** Casts hidden value to provided type, fails otherwise.
         * @return type-casted value
         */
        template <typename T>
        T& as() const
        {
            if (same_type<T>())
            {
                return *(reinterpret_cast<T*>(storage));
            }
            else
            {
                throw std::logic_error("Bad cast to " + demangledType<T>() +
                        " but the type is " + policy->type());
            }
        }

        /** @return true if type is same. */
        template <typename T>
        inline bool same_type() const
        {
            return (policy == owning_policy<T>()) || (policy == non_owning_policy<T>()) || same_type_fallback<T>();
        }

        /** @return true if type-id is same. */
        template <typename T>
        bool same_type_fallback() const
        {
            return policy->matches(typeid(T));
        }

        /** @return true if Any object is empty. */
        bool empty() const
        {
            return same_type<Empty>();
        }

    private:

        void assert_same_policy_type(BaseAnyPolicy* other_policy) {
            if (policy->policy_type() != other_policy->policy_type()) {
                throw std::logic_error("The policies are different: " +
                                       policy->policy_name() + " and " +
                                       other_policy->policy_name());
            }
        }

    private:
        BaseAnyPolicy* policy;
        void* storage;
    };

    inline bool operator==(const Any& lhs, const Any& rhs)
    {
        void* lhs_storage = lhs.storage;
        void* rhs_storage = rhs.storage;
        return lhs.policy == rhs.policy &&
            lhs.policy->equals(&lhs_storage, &rhs_storage);
    }

    inline bool operator!=(const Any& lhs, const Any& rhs)
    {
        return !(lhs == rhs);
    }

    /** Used to denote an empty Any object */
    struct Any::Empty
    {
        /** Equality operator */
        bool operator==(const Empty& other) const
        {
            return true;
        }
    };

    /** Erases value type i.e. converts it to Any
     * For input object of any type, it returns an Any object
     * which stores the input object's raw value. It saves the type
     * information internally to be recalled later by using recall_type().
     *
     * @param v value
     * @return Any object with the input value
     */
    template <typename T>
    inline Any erase_type(const T& v)
    {
        return Any(v);
    }

    template <typename T>
    inline Any erase_type_non_owning(T* v)
    {
        return Any(non_owning_policy<T>(), v);
    }

    /** Tries to recall Any type, fails when type is wrong.
     * Any stores type information of an object internally in a BaseAnyPolicy.
     * This function returns type-casted value if the internal type information
     * matches with the provided typename, otherwise throws std::logic_error.
     *
     * @param any object of Any
     * @return type-casted value
     */
    template <typename T>
    inline T recall_type(const Any& any)
    {
        return any.as<T>();
    }

}

#endif  //_ANY_H_
