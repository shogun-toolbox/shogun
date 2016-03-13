/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Soumyajit De
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

#ifndef MEAN_IMPL_H_
#define MEAN_IMPL_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/linalg/internal/Block.h>
#include <shogun/mathematics/linalg/internal/Sum.h>

#include <shogun/mathematics/eigen3.h>

#ifdef HAVE_VIENNACL
#include <shogun/mathematics/linalg/internal/Scale.h>
#include <shogun/mathematics/linalg/internal/opencl_util.h>
#include <shogun/lib/GPUMatrix.h>
#include <shogun/lib/GPUVector.h>

#endif

#include <string>

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
 * @brief Generic class mean which provides a static compute method. This class
 * is specialized for different types of matrices and backend, providing a means
 * to deal with various matrices directly without having to convert
 */
template <enum Backend,class Matrix>
struct mean
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/**
	 * Method that computes the mean of a dense matrix
	 *
	 * @param m the matrix whose mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the mean computed as \f$\mean_{i,j}m_{i,j}\f$
	 */
	static T compute(Matrix m, bool no_diag);

	/**
	 * Method that computes the mean of dense matrix blocks
	 *
	 * @param b the matrix-block whose mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the mean computed as \f$\mean_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<Matrix> b, bool no_diag);
};

/**
 * @brief Generic class mean symmetric which provides a static compute method. This class
 * is specialized for different types of matrices and backend, providing a means
 * to deal with various matrices directly without having to convert
 */
template <enum Backend,class Matrix>
struct mean_symmetric
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/**
	 * Method that computes the mean of a symmetric dense matrix
	 *
	 * @param m the matrix whose mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the mean computed as \f$\mean_{i,j}m_{i,j}\f$
	 */
	static T compute(Matrix m, bool no_diag);

	/**
	 * Method that computes the mean of symmetric dense matrix blocks
	 *
	 * @param b the matrix-block whose mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the mean computed as \f$\mean_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<Matrix> b, bool no_diag);
};

/**
 * @brief Generic class colwise_mean which provides a static compute method. This class
 * is specialized for different types of matrices and backend, providing a means
 * to deal with various matrices directly without having to convert
 */
template <enum Backend,class Matrix>
struct colwise_mean
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/**
	 * Method that computes column wise mean of a dense matrix
	 *
	 * @param m the matrix whose colwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the colwise mean computed as \f$s_j=\mean_{i}m_{i,j}\f$
	 */
	static SGVector<T> compute(Matrix m, bool no_diag);

	/**
	 * Method that computes column wise mean of dense matrix blocks
	 *
	 * @param b the matrix-block whose colwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the colwise mean computed as \f$s_j=\mean_{i}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<Matrix> b, bool no_diag);

	/**
	 * Method that computes the column wise mean of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose colwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(SGMatrix<T> m, SGVector<T> result, bool no_diag);

	/**
	 * Method that computes the column wise mean of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose colwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(Block<SGMatrix<T> > b, SGVector<T> result, bool no_diag);
};

/**
 * @brief Generic class rowwise_mean which provides a static compute method. This class
 * is specialized for different types of matrices and backend, providing a means
 * to deal with various matrices directly without having to convert
 */
template <enum Backend,class Matrix>
struct rowwise_mean
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/**
	 * Method that computes row wise mean of a dense matrix
	 *
	 * @param m the matrix whose rowwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the rowwise mean computed as \f$s_i=\mean_{j}m_{i,j}\f$
	 */
	static SGVector<T> compute(Matrix m, bool no_diag);

	/**
	 * Method that computes row wise mean of a dense matrix blocks
	 *
	 * @param b the matrix-block whose rowwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the rowwise mean computed as \f$s_i=\mean_{j}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<Matrix> b, bool no_diag);

	/**
	 * Method that computes the row wise mean of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose rowwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(SGMatrix<T> m, SGVector<T> result, bool no_diag);

	/**
	 * Method that computes the row wise mean of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose rowwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(Block<SGMatrix<T> > b, SGVector<T> result, bool no_diag);
};

/**
 * @brief Specialization of generic mean which works with SGMatrix and uses Eigen3
 * as backend for computing mean.
 */
template <class Matrix>
struct mean<Backend::EIGEN3,Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/**
	 * Method that computes the mean of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the mean computed as \f$\mean_{i,j}m_{i,j}\f$
	 */
	static T compute(SGMatrix<T> mat, bool no_diag)
	{
                REQUIRE(mat.num_rows > 0, "Matrix can not be empty!\n");
                
                // remove the main diagonal elements if required
                if (no_diag) 
                {
                    index_t len_major_diag = m.rows() < m.cols() ? m.rows() : m.cols();
                    return (sum<Backend::EIGEN3, SGMatrix<T> >::compute(mat, no_diag)
                    / (mat.num_rows * mat.num_cols - len_major_diag);
                }
                else
                {
                    return (sum<Backend::EIGEN3, SGMatrix<T> >::compute(mat, no_diag)
                    / (mat.num_rows * mat.num_cols);
                }
	}

	/**
	 * Method that computes the mean of SGMatrix blocks using Eigen3
	 *
	 * @param b the matrix-block whose mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the mean computed as \f$\mean_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<SGMatrix<T> > b, bool no_diag)
	{
		REQUIRE(b.m_row_size > 0, "Matrix can not be empty!\n");
                
                // remove the main diagonal elements if required
                if (no_diag) 
                {
                    index_t len_major_diag = b.m_row_size < b.m_col_size ? b.m_row_size : b.m_col_size;
                    return (sum<Backend::EIGEN3, Block<SGMatrix<T> > >::compute(b, no_diag)
                    / (b.m_row_size * b.m_col_size - len_major_diag);
                }
                else
                {
                    return (sum<Backend::EIGEN3, Block<SGMatrix<T> > >::compute(b, no_diag)
                    / (b.m_row_size * b.m_col_size);
                }
	}
};

/**
 * @brief Specialization of generic mean symmetric which works with SGMatrix and uses Eigen3
 * as backend for computing mean.
 */
template <class Matrix>
struct mean_symmetric<Backend::EIGEN3,Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/**
	 * Method that computes the mean of symmetric SGMatrix using Eigen3
	 *
	 * @param m the matrix whose mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the mean computed as \f$\mean_{i,j}m_{i,j}\f$
	 */
	static T compute(SGMatrix<T> mat, bool no_diag)
	{
                REQUIRE(mat.num_rows > 0, "Matrix can not be empty!\n");
                
                // remove the main diagonal elements if required
                if (no_diag) 
                {
                    return (sum_symmetricBackend::EIGEN3, SGMatrix<T> >
                    ::compute(mat, no_diag) / (mat.num_rows * (mat.num_cols - 1));
                }
                else
                {
                    return (sum_symmetric<Backend::EIGEN3, SGMatrix<T> >
                    ::compute(mat, no_diag) / (mat.num_rows * mat.num_cols);
                }
	}

	/**
	 * Method that computes the mean of symmetric SGMatrix blocks using Eigen3
	 *
	 * @param b the matrix-block whose mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the mean computed as \f$\mean_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<SGMatrix<T> > b, bool no_diag)
	{
		REQUIRE(b.m_row_size > 0, "Matrix can not be empty!\n");
                
                // remove the main diagonal elements if required
                if (no_diag) 
                {
                    return (sum_symmetric<Backend::EIGEN3, Block<SGMatrix<T> > >
                    ::compute(b, no_diag) / (b.m_row_size * (b.m_col_size - 1));
                }
                else
                {
                    return (sum_symmetric<Backend::EIGEN3, Block<SGMatrix<T> > >
                    ::compute(b, no_diag) / (b.m_row_size * b.m_col_size);
                }
	}
};

/**
 * @brief Specialization of generic colwise_mean which works with SGMatrix and uses Eigen3
 * as backend for computing mean.
 */
template <class Matrix>
struct colwise_mean<Backend::EIGEN3,Matrix> ////////////////// SGVector rowwise divide
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Return type */
	typedef SGVector<T> ReturnType;

	/** Eigen vector type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/**
	 * Method that computes the column wise mean of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose colwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the colwise mean computed as \f$s_j=\mean_{i}m_{i,j}\f$
	 */
	static SGVector<T> compute(SGMatrix<T> m, bool no_diag)
	{
		REQUIRE(m.num_cols > 0, "Matrix can not be empty!\n");
                SGVector<T> result(m.num_cols);
		compute(m, result, no_diag);
		return result;
	}

	/**
	 * Method that computes the column wise mean of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose colwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the colwise mean computed as \f$s_j=\mean_{i}b_{i,j}\f$
	 */
	static SGVector<T> compute(Block<SGMatrix<T> > b, bool no_diag)
	{
		REQUIRE(b.m_col_size > 0, "Matrix can not be empty!\n");
                SGVector<T> result(b.m_col_size);
		compute(b, result, no_diag);
		return result;
	}

	/**
	 * Method that computes the column wise mean of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose colwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(SGMatrix<T> mat, SGVector<T> result, bool no_diag)
	{
                Eigen::Map<VectorXt> r = result;
                r = colwise_sum<Backend::EIGEN3, SGMatrix<T> >::compute(mat, no_diag);
                
                // remove the main diagonal elements if required
                if (no_diag) 
                {
                        if (mat.num_cols >= mat.num_rows)
                        {
                                for (index_t i = 0; i < mat.num_rows; ++i)
                                        r[i] /= (mat.num_rows - 1);
                                for (index_t i = mat.num_rows; i < mat.num_cols; ++i)
                                        r[i] /= mat.num_rows;
                        }
                        else
                        {
                                for (index_t i = 0; i < mat.num_cols; ++i)
                                        r[i] /= (mat.num_rows - 1);
                                //scale<Backend::EIGEN3, Matrix>::compute(result, result, 1 / (mat.num_rows - 1));
                        }
                }
                else
                {
                        for (index_t i = 0; i < mat.num_cols; ++i)
                                r[i] /= mat.num_rows;
                        //scale<Backend::EIGEN3, Matrix>::compute(result, result, 1 / mat.num_rows);  
                }
	}

	/**
	 * Method that computes the column wise mean of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose colwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(Block<SGMatrix<T> > b, SGVector<T> result, bool no_diag)
	{
                Eigen::Map<VectorXt> r = result;
                r = colwise_sum<Backend::EIGEN3,Block<SGMatrix<T> > >::compute(b, no_diag);
                
                // remove the main diagonal elements if required
                if (no_diag) 
                {
                        if (b.m_col_size >= b.m_row_size)
                        {
                                for (index_t i = 0; i < b.m_row_size; ++i)
                                        r[i] /= (b.m_row_size - 1);
                                for (index_t i = b.m_row_size; i < b.m_col_size; ++i)
                                        r[i] /= b.m_row_size;
                        }
                        else
                        {
                                for (index_t i = 0; i < b.m_col_size; ++i)
                                        r[i] /= (b.m_row_size - 1);
                        }
                }
                else
                {
                        for (index_t i = 0; i < b.m_col_size; ++i)
                                r[i] /= b.m_row_size; 
                }
	}
};

/**
 * @brief Specialization of generic rowwise_mean which works with SGMatrix and uses Eigen3
 * as backend for computing mean.
 */
template <class Matrix>
struct rowwise_mean<Backend::EIGEN3,Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Return type */
	typedef SGVector<T> ReturnType;

	/** Eigen vector type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/**
	 * Method that computes the row wise mean of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose rowwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the rowwise mean computed as \f$s_i=\mean_{j}m_{i,j}\f$
	 */
	static SGVector<T> compute(SGMatrix<T> m, bool no_diag)
	{
		REQUIRE(m.num_rows > 0, "Matrix can not be empty!\n");
                SGVector<T> result(m.num_rows);
		compute(m, result, no_diag);
		return result;
	}

	/**
	 * Method that computes the row wise mean of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose rowwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the rowwise mean computed as \f$s_i=\mean_{j}m_{i,j}\f$
	 */
	static SGVector<T> compute(Block<SGMatrix<T> > b, bool no_diag)
	{
		REQUIRE(b.m_row_size > 0, "Matrix can not be empty!\n");
                SGVector<T> result(b.m_row_size);
		compute(b, result, no_diag);
		return result;
	}

	/**
	 * Method that computes the row wise mean of SGMatrix using Eigen3
	 *
	 * @param m the matrix whose rowwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(SGMatrix<T> mat, SGVector<T> result, bool no_diag)
	{
                Eigen::Map<VectorXt> r = result;
                r = rowwise_sum<Backend::EIGEN3, SGMatrix<T> >::compute(mat, no_diag);
                
                // remove the main diagonal elements if required
                if (no_diag) 
                {
                        if (mat.num_rows >= mat.num_cols)
                        {
                                for (index_t i = 0; i < mat.mat.num_cols; ++i)
                                        r[i] /= (mat.num_cols - 1);
                                for (index_t i = mat.num_cols; i < mat.num_rows; ++i)
                                        r[i] /= mat.num_cols;
                        }
                        else
                        {
                                for (index_t i = 0; i < mat.num_rows; ++i)
                                        r[i] /= (mat.num_cols - 1);
                        }
                }
                else
                {
                        for (index_t i = 0; i < mat.num_rows; ++i)
                                r[i] /= mat.num_cols;
                }
	}

	/**
	 * Method that computes the column wise mean of SGMatrix blocks
	 * using Eigen3
	 *
	 * @param b the matrix-block whose rowwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(Block<SGMatrix<T> > b, SGVector<T> result, bool no_diag)
	{
                Eigen::Map<VectorXt> r = result;
                r = colwise_sum<Backend::EIGEN3, Block<SGMatrix<T> > >::compute(b, no_diag);
                
                // remove the main diagonal elements if required
                if (no_diag) 
                {
                        if (b.m_row_size >= b.m_col_size)
                        {
                                for (index_t i = 0; i < b.m_col_size; ++i)
                                        r[i] /= (b.m_col_size - 1);
                                for (index_t i = b.m_col_size; i < b.m_row_size; ++i)
                                        r[i] /= b.m_col_size;
                        }
                        else
                        {
                                for (index_t i = 0; i < b.m_row_size; ++i)
                                        r[i] /= (b.m_col_size - 1);
                        }
                }
                else
                {
                        for (index_t i = 0; i < b.m_row_size; ++i)
                                r[i] /= b.m_col_size; 
                }
	}
};


#ifdef HAVE_VIENNACL
/**
 * @brief Specialization of generic mean which works with CGPUMatrix and uses ViennaCL
 * as backend for computing mean.
 */
template <class Matrix>
struct mean<Backend::VIENNACL,Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Generates the computation kernel */
	template <class T>
	static viennacl::ocl::kernel& generate_kernel(bool no_diag)
	{
		std::string kernel_name = "mean_" + ocl::get_type_string<T>();
		if (no_diag) kernel_name.append("_no_diag");

		if (ocl::kernel_exists(kernel_name))
			return ocl::get_kernel(kernel_name);

		std::string source = ocl::generate_kernel_preamble<T>(kernel_name);
		if (no_diag) source.append("#define NO_DIAG\n");

		source.append(
			R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* mat, int nrows, int ncols, int offset,
					__global DATATYPE* result)
				{
					__local DATATYPE buffer[WORK_GROUP_SIZE_1D];
					int size = nrows*ncols;

					int local_id = get_local_id(0);

					DATATYPE thread_mean = 0;
					for (int i=local_id; i<size; i+=WORK_GROUP_SIZE_1D)
					{
					#ifdef NO_DIAG
						if (!(i/nrows == i%nrows))
					#endif
						thread_mean += mat[i+offset];
					}

					buffer[local_id] = thread_mean;

					for (int j = WORK_GROUP_SIZE_1D/2; j > 0; j = j>>1)
					{
						barrier(CLK_LOCAL_MEM_FENCE);
						if (local_id < j)
							buffer[local_id] += buffer[local_id + j];
					}

					barrier(CLK_LOCAL_MEM_FENCE);

					if (get_global_id(0)==0)
						*result = buffer[0];
				}
			)"
		);

		viennacl::ocl::kernel& kernel = ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);
		kernel.global_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

	/**
	 * Method that computes the mean of CGPUMatrix using ViennaCL
	 *
	 * @param m the matrix whose mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the mean computed as \f$\mean_{i,j}m_{i,j}\f$
	 */
	static T compute(CGPUMatrix<T> mat, bool no_diag)
	{
		if (no_diag)
                {
                        REQUIRE(mat.num_rows != mat.num_cols, "Matrix is not square!\n");
                        return (sum<Backend::VIENNACL, CGPUMatrix<T> >
                            ::compute(mat, no_diag) / (mat.num_rows * (mat.num_cols - 1)));
                }
                else
                {
                        return (sum<Backend::VIENNACL, CGPUMatrix<T> >
                            ::compute(amt, no_diag) / (mat.num_rows * mat.num_cols);
                }
	}

	/**
	 * Method that computes the mean of CGPUMatrix blocks using ViennaCL
	 *
	 * @param b the matrix-block whose mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the mean computed as \f$\mean_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<CGPUMatrix<T> > b, bool no_diag)
	{
		SG_SERROR("The operation mean() on a matrix block is currently not supported\n");
		return 0;
	}
};

/**
 * @brief Specialization of generic mean symmetric which works with CGPUMatrix and uses ViennaCL
 * as backend for computing mean.
 */
template <class Matrix>
struct mean_symmetric<Backend::VIENNACL,Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/**
	 * Method that computes the mean of symmetric CGPUMatrix using ViennaCL
	 *
	 * @param m the matrix whose mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the mean computed as \f$\mean_{i,j}m_{i,j}\f$
	 */
	static T compute(CGPUMatrix<T> mat, bool no_diag)
	{
		return mean<Backend::VIENNACL, CGPUMatrix<T> >::compute(mat, no_diag);
	}

	/**
	 * Method that computes the mean of symmetric CGPUMatrix blocks using ViennaCL
	 *
	 * @param b the matrix-block whose mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the mean computed as \f$\mean_{i,j}b_{i,j}\f$
	 */
	static T compute(Block<CGPUMatrix<T> > b, bool no_diag)
	{
		SG_SERROR("The operation mean_symmetric() on a matrix block is currently not supported\n");
		return 0;
	}
};

/**
 * @brief Specialization of generic colwise_mean which works with CGPUMatrix and uses ViennaCL
 * as backend for computing mean.
 */
template <class Matrix>
struct colwise_mean<Backend::VIENNACL,Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Return type */
	typedef CGPUVector<T> ReturnType;

	/** Generates the computation kernel */
	template <class T>
	static viennacl::ocl::kernel& generate_kernel(bool no_diag)
	{
		std::string kernel_name = "colwise_mean_" + ocl::get_type_string<T>();
		if (no_diag) kernel_name.append("_no_diag");

		if (ocl::kernel_exists(kernel_name))
			return ocl::get_kernel(kernel_name);

		std::string source = ocl::generate_kernel_preamble<T>(kernel_name);
		if (no_diag) source.append("#define NO_DIAG\n");

		source.append(
			R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* mat, int nrows, int ncols, int offset,
					__global DATATYPE* result, int result_offset)
				{
					int j = get_global_id(0);

					if (j>=ncols)
						return;

					DATATYPE mean = 0;
					for (int i=0; i<nrows; i++)
					{
					#ifdef NO_DIAG
						if (i!=j)
					#endif
						mean += mat[offset+i+j*nrows];
					}

					result[j+result_offset] = mean;
				}
			)"
		);

		viennacl::ocl::kernel& kernel = ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

	/**
	 * Method that computes the column wise mean of CGPUMatrix using ViennaCL
	 *
	 * @param m the matrix whose colwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the colwise mean computed as \f$s_j=\mean_{i}m_{i,j}\f$
	 */
	static CGPUVector<T> compute(CGPUMatrix<T> m, bool no_diag)
	{
                CGPUVector<T> result(m.num_cols);
		compute(m, result, no_diag);
		return result;
	}

	/**
	 * Method that computes the column wise mean of CGPUMatrix blocks
	 * using ViennaCL
	 *
	 * @param b the matrix-block whose colwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the colwise mean computed as \f$s_j=\mean_{i}b_{i,j}\f$
	 */
	static CGPUVector<T> compute(Block<CGPUMatrix<T> > b, bool no_diag)
	{
		SG_SERROR("The operation colwise_mean() on a matrix block is currently not supported\n");
		return CGPUVector<T>();
	}

	/**
	 * Method that computes the column wise mean of CGPUMatrix using ViennaCL
	 *
	 * @param m the matrix whose colwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(CGPUMatrix<T> mat, CGPUVector<T> result, bool no_diag)
	{
                CGPUVector<T> sum_result(m.num_cols);
                sum_result = colwise_sum<Backend::VIENNACL, CGPUMatrix<T> >::compute(mat, no_diag);
                if (no_diag) 
                {
                        REQUIRE(mat.num_rows != mat.num_cols, "Matrix is not square!\n");
                        scale<Backend::VIENNACL, CGPUVector<T> >
                            ::compute(sum_result, result, 1 / (mat.num_rows - 1)); 
                }
                else
                {
                        scale<Backend::VIENNACL, CGPUVector<T> >
                            ::compute(sum_result, result, 1 / mat.num_rows);
                }  
	}

	/**
	 * Method that computes the column wise mean of CGPUMatrix blocks
	 * using ViennaCL
	 *
	 * @param b the matrix-block whose colwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(Block<CGPUMatrix<T> > b, CGPUVector<T> result, bool no_diag)
	{
		SG_SERROR("The operation colwise_mean() on a matrix block is currently not supported\n");
	}
};

/**
 * @brief Specialization of generic rowwise_mean which works with CGPUMatrix and uses ViennaCL
 * as backend for computing mean.
 */
template <class Matrix>
struct rowwise_mean<Backend::VIENNACL,Matrix>
{
	/** Scalar type */
	typedef typename Matrix::Scalar T;

	/** Return type */
	typedef CGPUVector<T> ReturnType;

	/** Generates the computation kernel */
	template <class T>
	static viennacl::ocl::kernel& generate_kernel(bool no_diag)
	{
		std::string kernel_name = "rowwise_mean_" + ocl::get_type_string<T>();
		if (no_diag) kernel_name.append("_no_diag");

		if (ocl::kernel_exists(kernel_name))
			return ocl::get_kernel(kernel_name);

		std::string source = ocl::generate_kernel_preamble<T>(kernel_name);
		if (no_diag) source.append("#define NO_DIAG\n");

		source.append(
			R"(
				__kernel void KERNEL_NAME(
					__global DATATYPE* mat, int nrows, int ncols, int offset,
					__global DATATYPE* result, int result_offset)
				{
					int i = get_global_id(0);

					if (i>=nrows)
						return;

					DATATYPE mean = 0;
					for (int j=0; j<ncols; j++)
					{
					#ifdef NO_DIAG
						if (i!=j)
					#endif
						mean += mat[offset+i+j*nrows];
					}

					result[i+result_offset] = mean;
				}
			)"
		);

		viennacl::ocl::kernel& kernel = ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);

		return kernel;
	}

	/**
	 * Method that computes the row wise mean of CGPUMatrix using ViennaCL
	 *
	 * @param m the matrix whose rowwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the rowwise mean computed as \f$s_i=\mean_{j}m_{i,j}\f$
	 */
	static CGPUVector<T> compute(CGPUMatrix<T> m, bool no_diag)
	{
                CGPUVector<T> result(m.num_rows);
		compute(m, result, no_diag);
		return result;
	}

	/**
	 * Method that computes the row wise mean of CGPUMatrix blocks
	 * using ViennaCL
	 *
	 * @param b the matrix-block whose rowwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @return the rowwise mean computed as \f$s_i=\mean_{j}m_{i,j}\f$
	 */
	static CGPUVector<T> compute(Block<CGPUMatrix<T> > b, bool no_diag)
	{
		SG_SERROR("The operation rowwise_mean() on a matrix block is currently not supported\n");
		return CGPUVector<T>();
	}

	/**
	 * Method that computes the row wise mean of CGPUMatrix using ViennaCL
	 *
	 * @param m the matrix whose rowwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(CGPUMatrix<T> mat, CGPUVector<T> result, bool no_diag)
	{
                CGPUVector<T> sum_result(m.num_rows);
                sum_result = rowwise_sum<Backend::VIENNACL, CGPUMatrix<T> >::compute(mat, no_diag);
                if (no_diag) 
                {
                        REQUIRE(mat.num_rows != mat.num_cols, "Matrix is not square!\n");
                        scale<Backend::VIENNACL, CGPUVector<T> >
                            ::compute(sum_result, result, 1 / (mat.num_cols - 1));
                }
                else
                {
                        scale<Backend::VIENNACL, CGPUVector<T> >
                            ::compute(sum_result, result, 1 / mat.num_cols;
                }
	}

	/**
	 * Method that computes the column wise mean of CGPUMatrix blocks
	 * using ViennaCL
	 *
	 * @param b the matrix-block whose rowwise mean has to be computed
	 * @param no_diag if true, diagonal entries are excluded from the mean
	 * @param result Pre-allocated vector for the result of the computation
	 */
	static void compute(Block<CGPUMatrix<T> > b, CGPUVector<T> result, bool no_diag)
	{
		SG_SERROR("The operation rowwise_mean() on a matrix block is currently not supported\n");
	}
};

#endif // HAVE_VIENNACL

}

}

}
#endif // mean_IMPL_H_
