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

#ifndef __LINALG_H__
#define __LINALG_H__

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/base/Parameter.h>
#include<shogun/mathematics/linalg/dotproduct/VectorDotProduct.h>

namespace shogun
{
/** Linear Algebra Backend options */
enum ELinAlgBackend
{
#ifdef HAVE_EIGEN3
Eigen3
#endif
};

/** Vector types available in shogun */
enum VectorTypes {PT_SGVector, PT_SGSparseVector};

/** @brief Linear Algebra class which contains global settings 
 *  for backends.
 */
class CLinearAlgebra : public CSGObject
{
public:
	/** Default Constructor for Linear Algebra class */
	CLinearAlgebra();

	/** Method to set backend for all linear algebra operations 
	 *  @param backend choice is Eigen3.
	 */
	void set_backend(ELinAlgBackend backend);

	/** Template getter for dot product computer */
	template <class T, class Vector>
	VectorDotProduct<T, Vector>* get_dot_computer();

	/** Destructor */
	~CLinearAlgebra();

	/** @return object name */
	virtual const char* get_name() const
	{
	    return "LinearAlgebra";
	}

private:
	/** Initialize with default values. The backend is set to Eigen3
         *  if eigen3 library is installed.
	 */
	void init();

	/** 2-d array of dot product computers */
	void ***dot_computers;
	
	/** Delete dot computer(call the destructor) */
	template<class T, class Vector>
	void delete_dot_computer();

	/** Delete all dot computers */
	void delete_all_dot_computers();
};

}
#endif //__LINALG_H__
