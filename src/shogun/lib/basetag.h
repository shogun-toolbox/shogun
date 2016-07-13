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

#ifndef _BASETAG_H_
#define _BASETAG_H_

#include <string>

namespace shogun
{
    /** @brief Base class for all tags.
     * This class stores name and not the type information for
     * a shogun object. It can be used as an identifier for a shogun object
     * where type information is not known.
     * One application of this can be found in CSGObject::set_param_with_btag().
     */
    class BaseTag
    {
    public:

        /** Constructor to initialize hash from name
         * @param _name name for tag
         */
        explicit BaseTag(const std::string& _name) : 
            m_name(_name), m_hash(std::hash<std::string>()(_name))
        {
        }

        /** Copy constructor
         * @param other basetag object to be copied
         */
        BaseTag(const BaseTag& other) : 
            m_name(other.m_name), m_hash(other.m_hash)
        {
        }

        /** Class Assignment operator
         * @param other basetag object to be assigned
         */
        BaseTag& operator=(const BaseTag& other)
        {
            m_name = other.m_name;
            m_hash = other.m_hash;
            return *this;
        }
        
        /** @return name of Tag */
        inline std::string name() const
        {
            return m_name;
        }

        /** @return hash of Tag */
        inline std::size_t hash() const
        {
            return m_hash;
        }

        /** Equality operator
         * @param first first BaseTag
         * @param second secondBaseTag
         */
        friend inline bool operator==(const BaseTag& first, const BaseTag& second);

        /** Inequality operator
         * @param first first BaseTag
         * @param second secondBaseTag
         */
        friend inline bool operator!=(const BaseTag& first, const BaseTag& second);

        /** Comparison operator
         * @param first first BaseTag
         * @param second secondBaseTag
         */
        friend inline bool operator<(const BaseTag& first, const BaseTag& second);

    private:
        /** name for object */
        std::string m_name;
        /** hash is stored for quick access from hash-map */
        size_t m_hash;
    };

    inline bool operator==(const BaseTag& first, const BaseTag& second)
    {
        return first.m_hash == second.m_hash ? first.m_name == second.m_name : false;
    }

    inline bool operator!=(const BaseTag& first, const BaseTag& second)
    {
        return !(first == second);
    }

    inline bool operator<(const BaseTag& first, const BaseTag& second)
    {
        return first.m_name < second.m_name;
    }

}

namespace std
{
    /** Overload hash for BaseTag */
    template <>
    struct hash<shogun::BaseTag>
    {
        std::size_t operator()(const shogun::BaseTag& basetag) const
        {
          return basetag.hash();
        }
    };

}

#endif  // _BASETAG_H_
