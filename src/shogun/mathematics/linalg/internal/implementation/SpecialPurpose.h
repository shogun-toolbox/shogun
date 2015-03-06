/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Khaled Nasr
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

#ifndef SPECIAL_PURPOSE_IMPL_H_
#define SPECIAL_PURPOSE_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUMatrix.h>
#include <shogun/mathematics/linalg/internal/opencl_util.h>
#endif // HAVE_VIENNACL

namespace shogun
{

namespace linalg
{

namespace implementation
{

namespace special_purpose
{

/** Generic class which is specialized for different backends to perform
 * the logistic operation
 */
template <enum Backend, class Matrix>
struct logistic
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Applies the elementwise logistic function f(x) = 1/(1+exp(-x)) to a matrix */
	static void compute(Matrix A, Matrix result);
};

#ifdef HAVE_EIGEN3

/** Specialization of logistic for the Eigen3 backend */
template <class Matrix>
struct logistic<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Applies the elementwise logistic function f(x) = 1/(1+exp(-x)) to a matrix */
	static void compute(SGMatrix<T> A, SGMatrix<T> result)
	{
		int32_t len = A.num_rows*A.num_cols;
		for (int32_t i=0; i<len; i++)
			result[i] = 1.0/(1+CMath::exp(-1*A[i]));
	}
};
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

/** Specialization of logistic for the ViennaCL backend */
template <class Matrix>
struct logistic<Backend::VIENNACL, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Applies the elementwise logistic function f(x) = 1/(1+exp(-x)) to a matrix */
	static void compute(CGPUMatrix<T> A, CGPUMatrix<T> result)
	{
		const std::string operation = "return 1.0/(1+exp(-1*element));";

		std::string kernel_name = "logistic_" + ocl::get_type_string<T>();
		viennacl::ocl::kernel& kernel =
			ocl::generate_single_arg_elementwise_kernel<T>(kernel_name, operation);

		kernel.global_work_size(0, ocl::align_to_multiple_1d(A.num_rows*A.num_cols));

		viennacl::ocl::enqueue(kernel(A.vcl_matrix(),
			cl_int(A.num_rows*A.num_cols), cl_int(A.offset),
			result.vcl_matrix(), cl_int(result.offset)));
	}
};

#endif // HAVE_VIENNACL

/** Generic class which is specialized for different backends to perform
 * the multiply_by_logistic_derivative operation
 */
template <enum Backend, class Matrix>
struct multiply_by_logistic_derivative
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Performs the operation C(i,j) = C(i,j) * A(i,j) * (1.0-A(i,j) for all i and j*/
	static void compute(Matrix A, Matrix C);
};

#ifdef HAVE_EIGEN3

/** Specialization of multiply_by_logistic_derivative for the Eigen3 backend */
template <class Matrix>
struct multiply_by_logistic_derivative<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Performs the operation C(i,j) = C(i,j) * A(i,j) * (1.0-A(i,j) for all i and j*/
	static void compute(SGMatrix<T> A, SGMatrix<T> C)
	{
		int32_t len = A.num_rows*A.num_cols;
		for (int32_t i=0; i<len; i++)
			C[i] *= A[i] * (1.0-A[i]);
	}
};
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

/** Specialization of multiply_by_logistic_derivative for the ViennaCL backend */
template <class Matrix>
struct multiply_by_logistic_derivative<Backend::VIENNACL, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Performs the operation C(i,j) = C(i,j) * A(i,j) * (1.0-A(i,j) for all i and j*/
	static void compute(CGPUMatrix<T> A, CGPUMatrix<T> C)
	{
		const std::string operation = "return element2 * element1*(1.0-element1);";

		std::string kernel_name = "multiply_by_logistic_derivative_" + ocl::get_type_string<T>();
		viennacl::ocl::kernel& kernel =
			ocl::generate_two_arg_elementwise_kernel<T>(kernel_name, operation);

		kernel.global_work_size(0, ocl::align_to_multiple_1d(A.num_rows*A.num_cols));

		viennacl::ocl::enqueue(kernel(
			A.vcl_matrix(), cl_int(A.num_rows*A.num_cols), cl_int(A.offset),
			C.vcl_matrix(), cl_int(C.offset),
			C.vcl_matrix(), cl_int(C.offset)));
	}
};

#endif // HAVE_VIENNACL

/** Generic class which is specialized for different backends to perform
 * the rectified_linear operation
 */
template <enum Backend, class Matrix>
struct rectified_linear
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Applies the elementwise rectified linear function f(x) = max(0,x) to a matrix */
	static void compute(Matrix A, Matrix result);
};

#ifdef HAVE_EIGEN3

/** Specialization of rectified_linear for the Eigen3 backend */
template <class Matrix>
struct rectified_linear<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Applies the elementwise rectified linear function f(x) = max(0,x) to a matrix */
	static void compute(SGMatrix<T> A, SGMatrix<T> result)
	{
		int32_t len = A.num_rows*A.num_cols;
		for (int32_t i=0; i<len; i++)
			result[i] = CMath::max((T)0, A[i]);
	}
};
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

/** Specialization of rectified_linear for the ViennaCL backend */
template <class Matrix>
struct rectified_linear<Backend::VIENNACL, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Applies the elementwise rectified linear function f(x) = max(0,x) to a matrix */
	static void compute(CGPUMatrix<T> A, CGPUMatrix<T> result)
	{
		const std::string operation = "return max((DATATYPE)0,element);";

		std::string kernel_name = "rectified_linear_" + ocl::get_type_string<T>();
		viennacl::ocl::kernel& kernel =
			ocl::generate_single_arg_elementwise_kernel<T>(kernel_name, operation);

		kernel.global_work_size(0, ocl::align_to_multiple_1d(A.num_rows*A.num_cols));

		viennacl::ocl::enqueue(kernel(A.vcl_matrix(),
			cl_int(A.num_rows*A.num_cols), cl_int(A.offset),
			result.vcl_matrix(), cl_int(result.offset)));
	}
};

#endif // HAVE_VIENNACL

/** Generic class which is specialized for different backends to perform
 * the multiply_by_rectified_linear_derivative operation
 */
template <enum Backend, class Matrix>
struct multiply_by_rectified_linear_derivative
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Performs the operation C(i,j) = C(i,j) * (A(i,j)!=0) for all i and j*/
	static void compute(Matrix A, Matrix C);
};

#ifdef HAVE_EIGEN3

/** Specialization of multiply_by_rectified_linear_derivative for the Eigen3 backend */
template <class Matrix>
struct multiply_by_rectified_linear_derivative<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Performs the operation C(i,j) = C(i,j) * (A(i,j)!=0) for all i and j*/
	static void compute(SGMatrix<T> A, SGMatrix<T> C)
	{
		int32_t len = A.num_rows*A.num_cols;
		for (int32_t i=0; i<len; i++)
			if (A[i]==0)
				C[i] = 0;
	}
};
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

/** Specialization of multiply_by_rectified_linear_derivative for the ViennaCL backend */
template <class Matrix>
struct multiply_by_rectified_linear_derivative<Backend::VIENNACL, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Performs the operation C(i,j) = C(i,j) * (A(i,j)!=0) for all i and j*/
	static void compute(CGPUMatrix<T> A, CGPUMatrix<T> C)
	{
		const std::string operation = "return element1==0 ? 0 : element2;";

		std::string kernel_name = "multiply_by_rectified_linear_derivative_" + ocl::get_type_string<T>();
		viennacl::ocl::kernel& kernel =
			ocl::generate_two_arg_elementwise_kernel<T>(kernel_name, operation);

		kernel.global_work_size(0, ocl::align_to_multiple_1d(A.num_rows*A.num_cols));

		viennacl::ocl::enqueue(kernel(
			A.vcl_matrix(), cl_int(A.num_rows*A.num_cols), cl_int(A.offset),
			C.vcl_matrix(), cl_int(C.offset),
			C.vcl_matrix(), cl_int(C.offset)));
	}
};

#endif // HAVE_VIENNACL

/** Applies the softmax function inplace to a matrix. The softmax function is
 * defined as \f$ f(A[i,j]) = \frac{exp(A[i,j])}{\sum_i exp(A[i,j])} \f$
 */
template <enum Backend, class Matrix>
struct softmax
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Applies the softmax function inplace to a matrix. The softmax function is
	 * defined as \f$ f(A[i,j]) = \frac{exp(A[i,j])}{\sum_i exp(A[i,j])} \f$
	 */
	static void compute(Matrix A);
};

#ifdef HAVE_EIGEN3

/** Specialization of softmax for the Eigen3 backend */
template <class Matrix>
struct softmax<Backend::EIGEN3, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Eigen matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/** Applies the softmax function inplace to a matrix. The softmax function is
	 * defined as \f$ f(A[i,j]) = \frac{exp(A[i,j])}{\sum_i exp(A[i,j])} \f$
	 */
	static void compute(SGMatrix<T> A)
	{
		Eigen::Map<MatrixXt> A_eig = A;

		float64_t max = A_eig.maxCoeff();

		for (int32_t j=0; j<A.num_cols; j++)
		{
			float64_t sum = 0;
			for (int32_t i=0; i<A.num_rows; i++)
				sum += CMath::exp(A(i,j)-max);

			float64_t normalizer = CMath::log(sum);
			for (int32_t k=0; k<A.num_rows; k++)
				A(k,j) = CMath::exp(A(k,j)-max-normalizer);
		}
	}
};
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

/** Specialization of softmax for the ViennaCL backend */
template <class Matrix>
struct softmax<Backend::VIENNACL, Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Generates the computation kernel */
	template <class T>
	static viennacl::ocl::kernel& generate_kernel()
	{
		std::string kernel_name = "softmax_" + ocl::get_type_string<T>();

		if (ocl::kernel_exists(kernel_name))
			return ocl::get_kernel(kernel_name);

		std::string source = ocl::generate_kernel_preamble<T>(kernel_name);

		source.append(
			R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* A, int nrows, int ncols, int offset)
				{
					int j = get_global_id(0);

					if (j>=ncols)
						return;

					DATATYPE col_max = -INFINITY;
					for (int i=0; i<nrows; i++)
						col_max = max(col_max, A[offset + i+j*nrows]);

					DATATYPE col_sum = 0;
					for (int i=0; i<nrows; i++)
						col_sum += exp(A[offset + i+j*nrows]-col_max);

					DATATYPE normalizer = log(col_sum);
					for (int i=0; i<nrows; i++)
					{
						int index = offset + i+j*nrows;
						A[index] = exp(A[index]-col_max-normalizer);
					}
				}
			)"
		);

		viennacl::ocl::kernel& kernel = ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

	/** Applies the softmax function inplace to a matrix. The softmax function is
	 * defined as \f$ f(A[i,j]) = \frac{exp(A[i,j])}{\sum_i exp(A[i,j])} \f$
	 */
	static void compute(CGPUMatrix<T> A)
	{
		viennacl::ocl::kernel& kernel = generate_kernel<T>();
		kernel.global_work_size(0, ocl::align_to_multiple_1d(A.num_cols));

		viennacl::ocl::enqueue(kernel(A.vcl_matrix(),
			cl_int(A.num_rows), cl_int(A.num_cols), cl_int(A.offset)));
	}
};

#endif // HAVE_VIENNACL

/** Generic class which is specialized for different backends to perform
 * the cross_entropy operation
 */
template <enum Backend,class Matrix>
struct cross_entropy
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Returns the cross entropy between P and Q. The cross entropy is defined as
	 * \f$ H(P,Q) = - \sum_{ij} P[i,j]log(Q[i,j]) \f$
	 */
	static T compute(Matrix P, Matrix Q);
};

#ifdef HAVE_EIGEN3
/** Specialization of cross_entropy for the Eigen3 backend */
template <class Matrix>
struct cross_entropy<Backend::EIGEN3,Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Eigen matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/** Returns the cross entropy between P and Q. The cross entropy is defined as
	 * \f$ H(P,Q) = - \sum_{ij} P[i,j]log(Q[i,j]) \f$
	 */
	static T compute(SGMatrix<T> P, SGMatrix<T> Q)
	{
		Eigen::Map<MatrixXt> P_eig = P;
		Eigen::Map<MatrixXt> Q_eig = Q;

		return -1*(P_eig.array() * (Q_eig.array()+1e-30).log()).sum();
	}
};
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
/** Specialization of cross_entropy for the ViennaCL backend */
template <class Matrix>
struct cross_entropy<Backend::VIENNACL,Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Generates the computation kernel */
	template <class T>
	static viennacl::ocl::kernel& generate_kernel()
	{
		std::string kernel_name = "cross_entropy_" + ocl::get_type_string<T>();

		if (ocl::kernel_exists(kernel_name))
			return ocl::get_kernel(kernel_name);

		std::string source = ocl::generate_kernel_preamble<T>(kernel_name);

		source.append(
			R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* p, int size, int p_offset,
					__global DATATYPE* q, int q_offset,
					__global DATATYPE* result)
				{
					__local DATATYPE buffer[WORK_GROUP_SIZE_1D];

					int local_id = get_local_id(0);

					DATATYPE thread_sum = 0;
					for (int i=local_id; i<size; i+=WORK_GROUP_SIZE_1D)
						thread_sum += p[i+p_offset]*log(q[i+q_offset]+1e-30);

					buffer[local_id] = thread_sum;

					for (int j = WORK_GROUP_SIZE_1D/2; j > 0; j = j>>1)
					{
						barrier(CLK_LOCAL_MEM_FENCE);
						if (local_id < j)
							buffer[local_id] += buffer[local_id + j];
					}

					barrier(CLK_LOCAL_MEM_FENCE);

					if (get_global_id(0)==0)
						*result = -1*buffer[0];
				}
			)"
		);

		viennacl::ocl::kernel& kernel = ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);
		kernel.global_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

	/** Returns the cross entropy between P and Q. The cross entropy is defined as
	 * \f$ H(P,Q) = - \sum_{ij} P[i,j]log(Q[i,j]) \f$
	 */
	static T compute(CGPUMatrix<T> P, CGPUMatrix<T> Q)
	{
		viennacl::ocl::kernel& kernel = generate_kernel<T>();

		CGPUVector<T> result(1);

		viennacl::ocl::enqueue(kernel(P.vcl_matrix(),
			cl_int(P.num_rows*P.num_cols), cl_int(P.offset),
			Q.vcl_matrix(), cl_int(Q.offset),
			result.vcl_vector()));

		return result[0];
	}
};
#endif // HAVE_VIENNACL

/** Generic class which is specialized for different backends to perform
 * the squared_error operation
 */
template <enum Backend,class Matrix>
struct squared_error
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Returns the squared error between P and Q. The squared error is defined as
	 * \f$ E(P,Q) = \frac{1}{2} \sum_{ij} (P[i,j]-Q[i,j])^2 \f$
	 */
	static T compute(Matrix P, Matrix Q);
};

#ifdef HAVE_EIGEN3
/** Specialization of squared_error for the Eigen3 backend */
template <class Matrix>
struct squared_error<Backend::EIGEN3,Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Eigen matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/** Returns the squared error between P and Q. The squared error is defined as
	 * \f$ E(P,Q) = \frac{1}{2} \sum_{ij} (P[i,j]-Q[i,j])^2 \f$
	 */
	static T compute(SGMatrix<T> P, SGMatrix<T> Q)
	{
		Eigen::Map<MatrixXt> P_eig = P;
		Eigen::Map<MatrixXt> Q_eig = Q;

		return 0.5 * (P_eig - Q_eig).array().square().sum();
	}
};
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
/** Specialization of squared_error for the ViennaCL backend */
template <class Matrix>
struct squared_error<Backend::VIENNACL,Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Generates the computation kernel */
	template <class T>
	static viennacl::ocl::kernel& generate_kernel()
	{
		std::string kernel_name = "squared_error_" + ocl::get_type_string<T>();

		if (ocl::kernel_exists(kernel_name))
			return ocl::get_kernel(kernel_name);

		std::string source = ocl::generate_kernel_preamble<T>(kernel_name);

		source.append(
			R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* p, int size, int p_offset,
					__global DATATYPE* q, int q_offset,
					__global DATATYPE* result)
				{
					__local DATATYPE buffer[WORK_GROUP_SIZE_1D];

					int local_id = get_local_id(0);

					DATATYPE thread_sum = 0;
					for (int i=local_id; i<size; i+=WORK_GROUP_SIZE_1D)
						thread_sum += pown(p[i+p_offset]-q[i+q_offset], 2);

					buffer[local_id] = thread_sum;

					for (int j = WORK_GROUP_SIZE_1D/2; j > 0; j = j>>1)
					{
						barrier(CLK_LOCAL_MEM_FENCE);
						if (local_id < j)
							buffer[local_id] += buffer[local_id + j];
					}

					barrier(CLK_LOCAL_MEM_FENCE);

					if (get_global_id(0)==0)
						*result = 0.5*buffer[0];
				}
			)"
		);

		viennacl::ocl::kernel& kernel = ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);
		kernel.global_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

	/** Returns the squared error between P and Q. The squared error is defined as
	 * \f$ E(P,Q) = \frac{1}{2} \sum_{ij} (P[i,j]-Q[i,j])^2 \f$
	 */
	static T compute(CGPUMatrix<T> P, CGPUMatrix<T> Q)
	{
		viennacl::ocl::kernel& kernel = generate_kernel<T>();

		CGPUVector<T> result(1);

		viennacl::ocl::enqueue(kernel(P.vcl_matrix(),
			cl_int(P.num_rows*P.num_cols), cl_int(P.offset),
			Q.vcl_matrix(), cl_int(Q.offset),
			result.vcl_vector()));

		return result[0];
	}
};
#endif // HAVE_VIENNACL

}

}

}

}
#endif // SPECIAL_PURPOSE_IMPL_H_
