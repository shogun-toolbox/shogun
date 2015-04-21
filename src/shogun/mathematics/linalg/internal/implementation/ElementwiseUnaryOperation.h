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

#ifndef ELEMENTWISE_OPERATION_H_
#define ELEMENTWISE_OPERATION_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUMatrix.h>
#include <shogun/mathematics/linalg/internal/opencl_config.h>
#include <shogun/mathematics/linalg/internal/opencl_util.h>
#endif // HAVE_VIENNACL

#include <algorithm>
#include <type_traits>

namespace shogun
{

namespace linalg
{

namespace implementation
{

/**
 * @brief Template struct elementwise_unary_operation. This struct is specialized for
 * computing element-wise operations for both matrices and vectors of CPU
 * (SGMatrix/SGVector) or GPU (CGPUMatrix/CGPUVector).
 */
template <enum Backend, class Operand, class ReturnType, class UnaryOp>
struct elementwise_unary_operation
{
};

/**
 * @brief Specialization for elementwise_unary_operation with NATIVE backend. It
 * provides two compute methods, one works for predefined unary operators and
 * the other one works with custom unary operators.
 *
 * The operand types MUST be of CPU types (SGMatrix/SGVector).
 */
template <class Operand, class ReturnType, class UnaryOp>
struct elementwise_unary_operation<Backend::NATIVE, Operand, ReturnType, UnaryOp>
{
	/** The scalar type of the operand */
	using T = typename Operand::Scalar;

	/** The scalar type of the result */
	using ST = typename ReturnType::Scalar;

	/** Ensure that this struct is not being instantiated with any GPU operand types */
	static_assert(!(std::is_same<CGPUMatrix<T>, Operand>::value
				|| std::is_same<CGPUVector<T>, Operand>::value),
			"NATIVE backend not allowed for GPU operands! Use SGMatrix/SGVector "
			"in order to use NATIVE or use VIENNACL backend instead.\n");

	/**
	 * Method compute that computes element-wise UnaryOp operation for the Operand.
	 * This method is used for predefined standard unary operators.
	 *
	 * @param operand The operand on which element-wise unary operation has to be performed
	 * @param result The result of applying the unary operator on each scalar of the operand
	 */
	static void compute(Operand operand, ReturnType result)
	{
		UnaryOp unary_op;
		compute(operand, result, unary_op);
	}

	/**
	 * Method compute that computes element-wise UnaryOp operation for the Operand.
	 * This method is used for custom unary operations, such as lambda expressions or
	 * function pointers as unary_op.
	 *
	 * @param operand The operand on which element-wise unary operation has to be performed
	 * @param result The result of applying the unary operator on each scalar of the operand
	 * @param unary_op The custom unary operator (a functor, lambda expression or a function
	 * pointer)
	 */
	static void compute(Operand operand, ReturnType result, UnaryOp unary_op)
	{
		static_assert(std::is_same<ST,decltype(unary_op(operand.data()[0]))>::value,
				"The return type of the unary operator and the scalar types of the "
				"result must be the same!\n");

		std::transform(operand.data(), operand.data()+operand.size(), result.data(),
		[&unary_op](T& value)
		{
			return unary_op(value);
		});
	}
};

#ifdef HAVE_EIGEN3
/**
 * @brief Specialization for elementwise_unary_operation with EIGEN3 backend. It
 * provides one compute method which work for predefined unary operators in Eigen3.
 *
 * The operand types MUST be other than GPU types (CGPUMatrix/CGPUVector).
 */
template <class Operand, class ReturnType, class UnaryOp>
struct elementwise_unary_operation<Backend::EIGEN3, Operand, ReturnType, UnaryOp>
{
	/** The scalar type of the operand */
	using T = typename Operand::Scalar;

	/** The scalar type of the result */
	using ST = typename UnaryOp::return_type;

	/** Ensure that this struct is not being instantiated with any GPU operand types */
	static_assert(!(std::is_same<CGPUMatrix<T>, Operand>::value
			 || std::is_same<CGPUVector<T>, Operand>::value),
			"EIGEN3 backend not allowed for GPU operands! Use SGMatrix/SGVector "
			"in order to use EIGEN3 or use VIENNACL backend instead.\n");

	/**
	 * Method compute that computes element-wise UnaryOp operation for the Operand
	 * using EIGEN3 backend.
	 *
	 * @param operand The operand on which element-wise unary operation has to be performed
	 * @param result The result of applying the unary operator on each scalar of the operand
	 */
	static void compute(Operand operand, ReturnType result)
	{
		UnaryOp unary_op;
		auto eigen_result=unary_op.compute_using_eigen3(operand);
		std::copy(eigen_result.data(), eigen_result.data()+eigen_result.size(), result.data());
	}
};
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
/**
 * @brief Specialization for elementwise_unary_operation with VIENNACL backend.
 * It provides one compute method which works for predefined unary operators.
 *
 * The operand types MUST be of GPU types (CGPUMatrix/CGPUVector).
 *
 * The return type and the operand type must be the same for ViennaCL.
 */
template <class Operand, class UnaryOp>
struct elementwise_unary_operation<Backend::VIENNACL, Operand, Operand, UnaryOp>
{
	/** The scalar type of the operand */
	using T = typename Operand::Scalar;

	/** Ensure that the scalar type is not a std::complex<double> type */
	static_assert(!std::is_same<T,complex128_t>::value,
			"Complex numbers not supported!\n");

	/** Ensure that this struct is being instantiated with only GPU operand types */
	static_assert(std::is_same<CGPUMatrix<T>, Operand>::value ||
			std::is_same<CGPUVector<T>, Operand>::value,
			"VIENNACL backend not allowed for CPU operands! Use CGPUMatrix/CGPUVector "
			"in order to use VIENNACL or use NATIVE/EIGEN3 backend instead.\n");

	/**
	 * Method compute that computes element-wise UnaryOp operation for the Operand
	 * using VIENNACL backend. Works for predefined unary operations.
	 *
	 * @param operand The operand on which element-wise unary operation has to be performed
	 * @param result The result of applying the unary operator on each scalar of the operand
	 */
	static void compute(Operand operand, Operand result)
	{
		static const std::string operation=UnaryOp::operation();
		static const std::string opname=UnaryOp::name();
		static const std::string kernel_name=opname+"_"+ocl::get_type_string<T>();

		viennacl::ocl::kernel& kernel=
			ocl::generate_single_arg_elementwise_kernel<T>(kernel_name, operation);

		kernel.global_work_size(0, ocl::align_to_multiple_1d(operand.size()));

		viennacl::ocl::enqueue(kernel(operand.data(),
			cl_int(operand.size()), cl_int(operand.offset),
			result.data(), cl_int(result.offset)));
	}
};

/**
 * @brief Specialization for elementwise_unary_operation with VIENNACL backend.
 * It provides one compute method which work for custom unary operators in VIENNACL.
 *
 * The operand types MUST be of GPU types (CGPUMatrix/CGPUVector).
 *
 * The return type and the operand type must be the same for ViennaCL.
 */
template <class Operand>
struct elementwise_unary_operation<Backend::VIENNACL, Operand, Operand, std::string>
{
	/** The scalar type of the operand */
	using T = typename Operand::Scalar;

	/** Ensure that the scalar type is not a std::complex<double> type */
	static_assert(!std::is_same<T,complex128_t>::value,
			"Complex numbers not supported!\n");

	/** Ensure that this struct is being instantiated with only GPU operand types */
	static_assert(std::is_same<CGPUMatrix<T>, Operand>::value ||
			std::is_same<CGPUVector<T>, Operand>::value,
			"VIENNACL backend not allowed for CPU operands! Use CGPUMatrix/CGPUVector "
			"in order to use VIENNACL or use NATIVE/EIGEN3 backend instead.\n");

	/**
	 * Method compute that computes element-wise UnaryOp operation for the Operand
	 * using VIENNACL backend. Works for custom unary operations.
	 *
	 * @param operand The operand on which element-wise unary operation has to be performed
	 * @param result The result of applying the unary operator on each scalar of the operand
	 * @param unary_op The operation body as string to be used inside an OpenCL kernel
	 */
	static void compute(Operand operand, Operand result, std::string unary_op)
	{
		std::hash<std::string> hash_fn;
		std::string hash=std::to_string(hash_fn(unary_op));
		std::string kernel_name="kernel_"+hash+"_"+ocl::get_type_string<T>();

		viennacl::ocl::kernel& kernel=
			ocl::generate_single_arg_elementwise_kernel<T>(kernel_name, unary_op);

		kernel.global_work_size(0, ocl::align_to_multiple_1d(operand.size()));

		viennacl::ocl::enqueue(kernel(operand.data(),
			cl_int(operand.size()), cl_int(operand.offset),
			result.data(), cl_int(result.offset)));
	}
};
#endif // HAVE_VIENNACL

}

}

}
#endif // ELEMENTWISE_OPERATION_H_
