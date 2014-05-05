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

#include<shogun/mathematics/linalg/global/LinearAlgebra.h>
#include<shogun/lib/DataType.h>
#include<shogun/lib/SGVector.h>
#include<shogun/mathematics/linalg/dotproduct/DenseEigen3DotProduct.h>

namespace shogun
{
CLinearAlgebra::CLinearAlgebra() : CSGObject()
{
    init();
    
    SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CLinearAlgebra::~CLinearAlgebra()
{
    delete_all_dot_computers();

	for(int i=0; i < 2; ++i)
    	    delete dot_computers[i];
	delete	dot_computers;
    
    SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

void CLinearAlgebra::init()
{
	dot_computers=NULL;
    #ifdef HAVE_EIGEN3
    set_backend(Eigen3);
    #endif
}

void CLinearAlgebra::set_backend(ELinAlgBackend linalg_backend)
{
    if(dot_computers)
    {    
		delete_all_dot_computers();
    }
    else
    {	
		dot_computers=new void**[2]();
    	for(int i=0; i < 2; ++i)
    	    dot_computers[i]=new void*[12];
    }

    switch (linalg_backend)
    {
	#ifdef HAVE_EIGEN3
        case Eigen3:
            dot_computers[PT_SGVector][PT_INT8]=
		new DenseEigen3DotProduct<int8_t>();
            dot_computers[PT_SGVector][PT_UINT8]=
		new DenseEigen3DotProduct<uint8_t>();
            dot_computers[PT_SGVector][PT_INT16]=
		new DenseEigen3DotProduct<int16_t>();
            dot_computers[PT_SGVector][PT_UINT16]=
		new DenseEigen3DotProduct<uint16_t>();
            dot_computers[PT_SGVector][PT_INT32]=
		new DenseEigen3DotProduct<int32_t>();
            dot_computers[PT_SGVector][PT_UINT32]=
		new DenseEigen3DotProduct<uint32_t>();
            dot_computers[PT_SGVector][PT_INT64]=
		new DenseEigen3DotProduct<int64_t>();
            dot_computers[PT_SGVector][PT_UINT64]=
		new DenseEigen3DotProduct<uint64_t>();
            dot_computers[PT_SGVector][PT_FLOAT32]=
		new DenseEigen3DotProduct<float32_t>();
            dot_computers[PT_SGVector][PT_FLOAT64]=
		new DenseEigen3DotProduct<float64_t>();
            dot_computers[PT_SGVector][PT_FLOATMAX]=
		new DenseEigen3DotProduct<floatmax_t>();
            dot_computers[PT_SGVector][PT_COMPLEX128]=
		new DenseEigen3DotProduct<complex128_t>();
	    break;
	#endif

        default:
            break;
    }

}

template<class T, class Vector>
VectorDotProduct<T, Vector>* CLinearAlgebra::get_dot_computer()
{
}

template<class T, class Vector>
void CLinearAlgebra::delete_dot_computer()
{
}

#define DOT_COMPUTER(T, Vector, PTYPE, VectorType) \
template<> \
void CLinearAlgebra::delete_dot_computer<T, Vector >() \
{\
delete static_cast<VectorDotProduct<T, Vector>*>(dot_computers[VectorType][PTYPE]); \
}\
template<> \
VectorDotProduct<T, Vector>* CLinearAlgebra::get_dot_computer<T, Vector >() \
{ \
return reinterpret_cast<VectorDotProduct<T, Vector >*>(dot_computers[VectorType][PTYPE]); \
}
DOT_COMPUTER(int8_t, SGVector<int8_t>, PT_INT8, PT_SGVector)
DOT_COMPUTER(uint8_t, SGVector<uint8_t>, PT_UINT8, PT_SGVector)
DOT_COMPUTER(int16_t, SGVector<int16_t>, PT_INT16, PT_SGVector)
DOT_COMPUTER(uint16_t, SGVector<uint16_t>, PT_UINT16, PT_SGVector)
DOT_COMPUTER(int32_t, SGVector<int32_t>, PT_INT32, PT_SGVector)
DOT_COMPUTER(uint32_t, SGVector<uint32_t>, PT_UINT32, PT_SGVector)
DOT_COMPUTER(int64_t, SGVector<int64_t>, PT_INT64, PT_SGVector)
DOT_COMPUTER(uint64_t, SGVector<uint64_t>, PT_UINT64, PT_SGVector)
DOT_COMPUTER(float32_t, SGVector<float32_t>, PT_FLOAT32, PT_SGVector)
DOT_COMPUTER(float64_t, SGVector<float64_t>, PT_FLOAT64, PT_SGVector)
DOT_COMPUTER(floatmax_t, SGVector<floatmax_t>, PT_FLOATMAX, PT_SGVector)
DOT_COMPUTER(complex128_t, SGVector<complex128_t>, PT_COMPLEX128, PT_SGVector)

void CLinearAlgebra::delete_all_dot_computers()
{
	delete_dot_computer<int8_t, SGVector<int8_t> >();
	delete_dot_computer<uint8_t, SGVector<uint8_t> >();
	delete_dot_computer<int16_t, SGVector<int16_t> >();
	delete_dot_computer<uint16_t, SGVector<uint16_t> >();
	delete_dot_computer<int32_t, SGVector<int32_t> >();
	delete_dot_computer<uint32_t, SGVector<uint32_t> >();
	delete_dot_computer<int64_t, SGVector<int64_t> >();
	delete_dot_computer<uint64_t, SGVector<uint64_t> >();
	delete_dot_computer<float32_t, SGVector<float32_t> >();
	delete_dot_computer<float64_t, SGVector<float64_t> >();
	delete_dot_computer<floatmax_t, SGVector<floatmax_t> >();
	delete_dot_computer<complex128_t, SGVector<complex128_t> >();
}
}
