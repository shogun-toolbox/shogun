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

#ifndef EIGEN3_DOT_PRODUCT_H_
#define EIGEN3_DOT_PRODUCT_H_

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/linalg/dotproduct/VectorDotProduct.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>

using namespace Eigen;

namespace shogun
{
/** @brief template class for Eigen3 dot product that performs dot product
 *  operation using Eigen3 Library
 */
template <class T> class CDenseEigen3DotProduct : public CVectorDotProduct<T, SGVector<T> >
{
public:
    /** constructor */
    CDenseEigen3DotProduct() : CVectorDotProduct<T, SGVector<T> >()
    {
        CSGObject::set_generic<T>();
    }

    /** compute method which performs dot product operation
     *  @param vector1 input dense vector
     *  @param vector2 input dense vector
     *  @return the result
     */
    virtual T compute(SGVector<T> vector1, SGVector<T> vector2) const;

    /** destructor */
    virtual ~CDenseEigen3DotProduct()
    {
    }

    /** @return object name */
    virtual const char* get_name() const
    {
        return "DenseEigen3DotProduct";
    }
};
}
#endif //EIGEN3_DOT_PRODUCT_H_
#endif //HAVE_EIGEN3
