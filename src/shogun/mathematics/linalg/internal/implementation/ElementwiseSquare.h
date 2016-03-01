/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
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

#ifndef ELEMENTWISESQUARE_IMPL_H_
#define ELEMENTWISESQUARE_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/linalg/internal/Block.h>
#include <shogun/io/SGIO.h>

#include <shogun/mathematics/eigen3.h>

#ifdef HAVE_VIENNACL
#include <shogun/mathematics/linalg/internal/opencl_util.h>
#include <shogun/lib/GPUMatrix.h>
#endif

namespace shogun
{

namespace linalg
{

/**
 * All backend specific implementations are defined within this namespace
 */
namespace implementation
{

/**
 * @brief Generic class square which provides a static compute method. This class
 * is specialized for different types of matrices and backend, providing a mean
 * to deal with various matrices directly without having to convert
 */
template <enum Backend,class Matrix>
struct elementwise_square
{
	/**
	 * Method that computes the square of co-efficients of a dense matrix
	 *
	 * @param m the matrix whose squared co-efficients matrix has to be computed
	 * @return another matrix whose co-efficients are \f$m'_{i,j}=m_(i,j}^2\f$
	 * for all \f$i,j\f$
	 */
	static Matrix compute(Matrix m);

	/**
	 * Method that computes the square of co-efficients of a dense matrix-block
	 *
	 * @param b the matrix-block whose squared co-efficients matrix has to be computed
	 * @return another matrix whose co-efficients are \f$m'_{i,j}=b_(i,j}^2\f$
	 * for all \f$i,j\f$
	 */
	static Matrix compute(Block<Matrix> b);
};

/**
 * @brief Partial specialization of generic elementwise_square for the Eigen3 backend
 */
template <class Matrix>
struct elementwise_square<Backend::EIGEN3,Matrix>
{
	/** The scalar type */
	typedef typename Matrix::Scalar T;

	/** The return type */
	typedef SGMatrix<T> ReturnType;

	/** Eigen3 matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/**
	 * Method that computes the square of co-efficients of a dense matrix
	 *
	 * @param m the matrix whose squared co-efficients matrix has to be computed
	 * @return another matrix whose co-efficients are \f$m'_{i,j}=m_(i,j}^2\f$
	 * for all \f$i,j\f$
	 */
	static SGMatrix<T> compute(SGMatrix<T> m)
	{
		SGMatrix<T> result(m.num_rows, m.num_cols);
		compute(m, result);
		return result;
	}

	/**
	 * Method that computes the square of co-efficients of a dense matrix-block
	 *
	 * @param b the matrix-block whose squared co-efficients matrix has to be computed
	 * @return another matrix whose co-efficients are \f$m'_{i,j}=b_(i,j}^2\f$
	 * for all \f$i,j\f$
	 */
	static SGMatrix<T> compute(Block<SGMatrix<T> > b)
	{
		SGMatrix<T> result(b.m_row_size, b.m_col_size);
		compute(b, result);
		return result;
	}

	/**
	 * Method that computes the square of co-efficients of a dense matrix
	 *
	 * @param m the matrix whose squared co-efficients matrix has to be computed
	 * @param result Pre-allocated matrix for the result of the computation
	 */
	static void compute(SGMatrix<T> mat, SGMatrix<T> result)
	{
		Eigen::Map<MatrixXt> m = mat;
		Eigen::Map<MatrixXt> r = result;

		r = m.array().template square();
	}

	/**
	 * Method that computes the square of co-efficients of a dense matrix-block
	 *
	 * @param b the matrix-block whose squared co-efficients matrix has to be computed
	 * @param result Pre-allocated matrix for the result of the computation
	 */
	static void compute(Block<SGMatrix<T> > b, SGMatrix<T> result)
	{
		Eigen::Map<MatrixXt> map = b.m_matrix;
		Eigen::Map<MatrixXt> r = result;

		Eigen::Block< Eigen::Map<MatrixXt> > m = map.block(
			b.m_row_begin, b.m_col_begin,
			b.m_row_size, b.m_col_size);

		r = m.array().template square();
	}
};


#ifdef HAVE_VIENNACL
/**
 * @brief Partial specialization of generic elementwise_square for the ViennaCL backend
 */
template <class Matrix>
struct elementwise_square<Backend::VIENNACL,Matrix>
{
	/** The scalar type */
	typedef typename Matrix::Scalar T;

	/** The return type */
	typedef CGPUMatrix<T> ReturnType;

	/**
	 * Method that computes the square of co-efficients of a dense matrix
	 *
	 * @param m the matrix whose squared co-efficients matrix has to be computed
	 * @return another matrix whose co-efficients are \f$m'_{i,j}=m_(i,j}^2\f$
	 * for all \f$i,j\f$
	 */
	static CGPUMatrix<T> compute(CGPUMatrix<T> m)
	{
		CGPUMatrix<T> result(m.num_rows, m.num_cols);
		compute(m, result);
		return result;
	}

	/**
	 * Method that computes the square of co-efficients of a dense matrix-block
	 *
	 * @param b the matrix-block whose squared co-efficients matrix has to be computed
	 * @return another matrix whose co-efficients are \f$m'_{i,j}=b_(i,j}^2\f$
	 * for all \f$i,j\f$
	 */
	static CGPUMatrix<T> compute(Block<CGPUMatrix<T> > b)
	{
		SG_SERROR("The operation elementwise_square() on a matrix block is currently not supported\n");
		return CGPUMatrix<T>();
	}

	/**
	 * Method that computes the square of co-efficients of a dense matrix
	 *
	 * @param m the matrix whose squared co-efficients matrix has to be computed
	 * @param result Pre-allocated matrix for the result of the computation
	 */
	static void compute(CGPUMatrix<T> mat, CGPUMatrix<T> result)
	{
		const std::string operation = "return element*element;";

		std::string kernel_name = "elementwise_square_" + ocl::get_type_string<T>();
		viennacl::ocl::kernel& kernel =
			ocl::generate_single_arg_elementwise_kernel<T>(kernel_name, operation);

		kernel.global_work_size(0, ocl::align_to_multiple_1d(mat.num_rows*mat.num_cols));

		viennacl::ocl::enqueue(kernel(mat.vcl_matrix(),
			cl_int(mat.num_rows*mat.num_cols), cl_int(mat.offset),
			result.vcl_matrix(), cl_int(result.offset)));
	}

	/**
	 * Method that computes the square of co-efficients of a dense matrix-block
	 *
	 * @param b the matrix-block whose squared co-efficients matrix has to be computed
	 * @param result Pre-allocated matrix for the result of the computation
	 */
	static void compute(Block<SGMatrix<T> > b, SGMatrix<T> result)
	{
		SG_SERROR("The operation elementwise_square() on a matrix block is currently not supported\n");
	}
};

#endif // HAVE_VIENNACL

}

}

}
#endif // ELEMENTWISESQUARE_IMPL_H_
