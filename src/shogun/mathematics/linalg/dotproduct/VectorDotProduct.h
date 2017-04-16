/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Written (W) 2014 Sunil K. Mahendrakar
 */

#ifndef VECTOR_DOT_PRODUCT_H_
#define VECTOR_DOT_PRODUCT_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>

namespace shogun
{
/** @brief Abstract template base class for vector dot product computers */
template <class T, class Vector >
class CVectorDotProduct : public CSGObject
{
public:
    /** constructor */
    CVectorDotProduct() : CSGObject()
    {
        SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
    }

    /** compute method for computing dot product of two vectors
     *  @param vector1 input vector
     *  @param vector2 input vector
     *  @return the result of dot product operation
     */
    virtual T compute(Vector vector1, Vector vector2) const = 0;

    /** destructor */
    virtual ~CVectorDotProduct()
    {
        SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
    }

    /** @return object name */
    virtual const char* get_name() const
    {
        return "VectorDotProduct";
    }
};
}
#endif //VECTOR_DOT_PRODUCT_H_
