/*
 * Copyright (c) 2014, Shogun Toolbox Foundation
 * All rights reserved.
 *
 * Written (W) 2014 Sunil K. Mahendrakar
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
 */

#ifndef DENSE_VECTOR_DOT_OPERATOR_H_
#define DENSE_VECTOR_DOT_OPERATOR_H_

#include <shogun/lib/config.h>
#include<shogun/mathematics/linalg/global/LinearAlgebra.h>
#include<shogun/mathematics/linalg/linop/LinearOperator.h>

namespace shogun
{

/** @brief Abstract template base class that represents a Vector Dot operator
 */
template <class T, class Vector>
class VectorDotOperator : public CLinearOperator<T, Vector>
{
public:
	/** Default Constructor
	*  @param vec input vector on which the dot operator can apply
	*/
	VectorDotOperator(Vector vec) 
	: CLinearOperator<T, Vector>(), vector(vec)
	{
		linalg=new CLinearAlgebra();
		SG_REF(linalg);		
		
		SG_SGCDEBUG("%s created (%p)\n", this->get_name(), this)
	}

	/** Abstract method that applies the dot operator to a vector
     	* (\f$\sum_{i=1}^d a_ib_i$ where $a,b$ are $d$-dimensional vectors)
	* @param other the operand vector to which the dot operator applies
	* @return the result
	*/
	virtual T apply(Vector other) const
	{
		return linalg->get_dot_computer<T, Vector>()->compute(vector, other);
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "VectorDotOperator";
	}

	/** Destructor */
	virtual ~VectorDotOperator()
	{
		SG_UNREF(linalg);

    	SG_SGCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
	}

private:
	/** The input vector on which the dot operator can apply */
	Vector vector;	
	
	/** The Linear Algebra class */
	CLinearAlgebra* linalg;
};

}
#endif //DENSE_VECTOR_DOT_OPERATOR_H_
