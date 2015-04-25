/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef ELEMENTWISE_OPERATIONS_H_
#define ELEMENTWISE_OPERATIONS_H_

#include <shogun/mathematics/linalg/internal/implementation/operations/Sin.h>
#include <shogun/mathematics/linalg/internal/implementation/operations/opencl_operation.h>
#include <shogun/mathematics/linalg/internal/implementation/util/AllocResultUtil.h>
#include <shogun/mathematics/linalg/internal/implementation/ElementwiseUnaryOperation.h>

namespace shogun
{

namespace linalg
{

/**
 * Template method for computing custom unary operations element-wise for matrices
 * and vectors using NATIVE backend. Works for SGMatrix/SGVector.
 *
 * This method returns the result in a newly allocated matrix/vector.
 *
 * @param operand The operand on which the element-wise operation has to be performed
 * @param unary_op The custom unary operator
 * @return The result of the unary operator applied element-wise on the operand
 */
template <class Operand, class UnaryOp>
auto elementwise_compute(Operand operand, UnaryOp unary_op)
-> typename Operand::template container_type<decltype(unary_op(operand.data()[0]))>
{
	typedef decltype(unary_op(operand.data()[0])) ST;
	typedef typename Operand::template container_type<ST> ReturnType;

	ReturnType result=util::allocate_result<Operand,ReturnType>::alloc(operand);

	implementation::elementwise_unary_operation<Backend::NATIVE, Operand,
		ReturnType, UnaryOp>::compute(operand, result, unary_op);

	return result;
}

/**
 * Template method for computing custom unary operations element-wise for matrices
 * and vectors using NATIVE backend. Works for SGMatrix/SGVector.
 *
 * This method computes the result in-place.
 *
 * @param operand The operand on which the element-wise operation has to be performed
 * @param unary_op The custom unary operator
 */
template <class Operand, class UnaryOp>
void elementwise_compute_inplace(Operand operand, UnaryOp unary_op)
{
	typedef typename Operand::Scalar T;
	typedef decltype(unary_op(operand.data()[0])) ST;
	static_assert(std::is_same<T,ST>::value, "Scalar type mismatch!\n");

	implementation::elementwise_unary_operation<Backend::NATIVE, Operand,
		Operand, UnaryOp>::compute(operand, operand, unary_op);
}

#ifdef HAVE_VIENNACL
/**
 * Template method for computing custom unary operations element-wise for matrices
 * and vectors using VIENNACL/OPENCL backend. Works for CGPUMatrix/CGPUVector.
 *
 * This method returns the result in a newly allocated matrix/vector.
 *
 * @param operand The operand on which the element-wise operation has to be performed
 * @param unary_op The custom unary operator string
 * @return The result of the unary operator applied element-wise on the operand
 */
template <class Operand>
Operand elementwise_compute(Operand operand, std::string unary_op)
{
	Operand result=util::allocate_result<Operand,Operand>::alloc(operand);
	operations::ocl_operation operation(unary_op);

	implementation::elementwise_unary_operation<Backend::VIENNACL, Operand,
		Operand, operations::ocl_operation>::compute(operand, result, operation);

	return result;
}

/**
 * Template method for computing custom unary operations element-wise for matrices
 * and vectors using VIENNACL/OPENCL backend. Works for CGPUMatrix/CGPUVector.
 *
 * This method computes the result in-place.
 *
 * @param operand The operand on which the element-wise operation has to be performed
 * @param unary_op The custom unary operator string
 */
template <class Operand>
void elementwise_compute_inplace(Operand operand, std::string unary_op)
{
	operations::ocl_operation operation(unary_op);
	implementation::elementwise_unary_operation<Backend::VIENNACL, Operand,
		Operand, operations::ocl_operation>::compute(operand, operand, operation);
}
#endif // HAVE_VIENNACL

/**
 * Template method for computing element-wise sin for matrices and vectors.
 *
 * This method returns the result in a newly allocated matrix/vector.
 *
 * @param operand The operand on which the element-wise operation has to be performed
 * @return The result of the unary operator applied element-wise on the operand
 */
template <Backend backend, class Operand>
typename Operand::template container_type<typename operations::sin<typename Operand::Scalar>::return_type>
elementwise_sin(Operand operand)
{
	typedef typename Operand::Scalar T;
	typedef typename operations::sin<T>::return_type ST;
	typedef typename Operand::template container_type<ST> ReturnType;

	ReturnType result=util::allocate_result<Operand,ReturnType>::alloc(operand);

	operations::sin<T> operation;
	implementation::elementwise_unary_operation<backend, Operand,
		ReturnType, operations::sin<T>>::compute(operand, result, operation);

	return result;
}

/**
 * Template method for computing element-wise sin for matrices and vectors.
 *
 * This method computes the result in-place.
 *
 * @param operand The operand on which the element-wise operation has to be performed
 * @return The result of the unary operator applied element-wise on the operand
 */
template <Backend backend, class Operand>
void elementwise_sin_inplace(Operand operand)
{
	typedef typename Operand::Scalar T;
	typedef typename operations::sin<T>::return_type ST;
	static_assert(std::is_same<T,ST>::value, "Scalar type mismatch!\n");

	operations::sin<T> operation;
	implementation::elementwise_unary_operation<backend, Operand,
		Operand, operations::sin<T>>::compute(operand, operand, operation);
}

}

}
#endif // ELEMENTWISE_OPERATIONS_H_
