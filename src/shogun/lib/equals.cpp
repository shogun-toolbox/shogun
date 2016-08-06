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
 */

#include <shogun/lib/common.h>
#include <shogun/lib/equals.h>

#include <shogun/base/SGObject.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{
    #define BASIC_EQUALS(type)                  \
        template <>                             \
        bool equals(type* lhs, type* rhs)       \
        {                                       \
            return *lhs == *rhs;                \
        }

    BASIC_EQUALS(bool);
    BASIC_EQUALS(char);
    BASIC_EQUALS(int8_t);
    BASIC_EQUALS(uint8_t);
    BASIC_EQUALS(int16_t);
    BASIC_EQUALS(uint16_t);
    BASIC_EQUALS(int32_t);
    BASIC_EQUALS(uint32_t);
    BASIC_EQUALS(int64_t);
    BASIC_EQUALS(uint64_t);
    BASIC_EQUALS(float32_t);
    BASIC_EQUALS(float64_t);
    BASIC_EQUALS(floatmax_t);
    BASIC_EQUALS(complex128_t);

    template <>
    bool equals(CSGObject** lhs, CSGObject** rhs)
    {
        return (*lhs)->equals(*rhs);
    }

    template <>
    bool equals(CKernel** lhs, CKernel** rhs)
    {
        return (*lhs)->equals(*rhs);
    }

}

#undef BASIC_EQUALS
