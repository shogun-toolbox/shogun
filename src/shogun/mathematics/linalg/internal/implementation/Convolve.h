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

#ifndef CONVOLVE_H_
#define CONVOLVE_H_

#include <shogun/lib/config.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/neuralnets/NeuralLayer.h>

#include <shogun/io/SGIO.h>

#include <shogun/mathematics/eigen3.h>

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

/** Generic class which is specialized for different backends to perform
 * the convolve operation
 */
template <enum Backend, class Matrix>
struct convolve
{
	/** The scalar type */
	typedef typename Matrix::Scalar T;

	/** Computes the 2D convolution of X with W
	 *
	 * NOTE: For the ViennaCL backend, the size of W (number of bytes) must not exceed
	 * [CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE](http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html).
	 *
	 * @param X Input image
	 * @param W Filter coefficients. The dimensions of the matrix must be odd-numbered.
	 * @param Y Output image of the same size as the input image, as the borders
	 * of the input image are implicitly padded with zeros during the computation
	 * @param flip If true the filter coefficients are flipped, performing cross-correlation
	 * instead of convolution
	 * @param overwrite If true, the values in Y are overwritten with result of the
	 * computation. Otherwise, the result is added to the existing values in Y.
	 * @param stride_x Stride in the x (column) direction
	 * @param stride_y Stride in the y (row) direction
	 */
	static void compute(Matrix X, Matrix W, Matrix Y, bool flip ,
		bool overwrite, int32_t stride_x, int32_t stride_y, ENLAutoencoderPosition autoencoder_position = NLAP_NONE);
};


/** Partial specialization of convolve for the Eigen3 backend */
template <class Matrix>
struct convolve<Backend::EIGEN3, Matrix>
{
	/** The scalar type */
	typedef typename Matrix::Scalar T;

	/** Eigen3 matrix type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> MatrixXt;

	/** Eigen3 vector type */
	typedef Eigen::Matrix<T,Eigen::Dynamic,1> VectorXt;

	/** Computes the 2D convolution of X with W
	 *
	 * @param X Input image
	 * @param W Filter coefficients. The dimensions of the matrix must be odd-numbered.
	 * @param Y Output image of the same size as the input image, as the borders
	 * of the input image are implicitly padded with zeros during the computation
	 * @param flip If true the filter coefficients are flipped, performing cross-correlation
	 * instead of convolution
	 * @param overwrite If true, the values in Y are overwritten with result of the
	 * computation. Otherwise, the result is added to the existing values in Y.
	 * @param stride_x Stride in the x (column) direction
	 * @param stride_y Stride in the y (row) direction
	 */
	static void compute(SGMatrix<T> X, SGMatrix<T> W, SGMatrix<T> Y, bool flip ,
		bool overwrite, int32_t stride_x, int32_t stride_y, ENLAutoencoderPosition autoencoder_position = NLAP_NONE)
	{
		int32_t width = X.num_cols;
		int32_t height = X.num_rows;

		int32_t kx = W.num_cols;
		int32_t ky = W.num_rows;

		int32_t rx = (kx-1)/2;
		int32_t ry = (ky-1)/2;

		for (int32_t x=0; x<width; x+=stride_x)
		{
			int32_t xout = autoencoder_position == NLAP_NONE ? x/stride_x : x;

			for (int32_t y=0; y<height; y+=stride_y)
			{
				int32_t yout = autoencoder_position == NLAP_NONE ? y/stride_y : y;

				T sum = overwrite ? 0 : Y(yout,xout);
				for (int32_t x1=x-rx; x1<=x+rx; x1++)
				{
					int32_t wx = flip ? x1-x+rx : rx-x1+x;
					for (int32_t y1=y-ry; y1<=y+ry; y1++)
					{
						if (x1>=0 && y1>=0 && x1<width && y1<height)
						{
							if (flip)
								sum += W(y1-y+ry,wx)*X(y1,x1);
							else
								sum += W(ry-y1+y,wx)*X(y1,x1);
						}
					}
				}
				Y(yout,xout) = sum;
			}
		}
	}
};

#ifdef HAVE_VIENNACL

/** Partial specialization of convolve for the ViennaCL backend */
template <class Matrix>
struct convolve<Backend::VIENNACL, Matrix>
{
	/** The scalar type */
	typedef typename Matrix::Scalar T;

	/** Generates the computation kernel for convolution with a stride of 1*/
	template <class T>
	static viennacl::ocl::kernel& generate_kernel_unity_stride(
		int32_t radius_x, int32_t radius_y, bool flip, bool overwrite)
	{
		std::string kernel_name =
			"convolve_unity_stride_" + ocl::get_type_string<T>() + "_" +
			std::to_string(radius_x) + "_" + std::to_string(radius_y);

		if (flip) kernel_name.append("_flip");
		if (overwrite) kernel_name.append("_overwrite");

		if (ocl::kernel_exists(kernel_name))
			return ocl::get_kernel(kernel_name);

		std::string source = ocl::generate_kernel_preamble<T>(kernel_name);

		if (flip) source.append("#define FLIP\n");
		if (overwrite) source.append("#define OVERWRITE\n");

		source.append("#define RADIUS_X " + std::to_string(radius_x) + "\n");
		source.append("#define RADIUS_Y " + std::to_string(radius_y) + "\n");

		source.append(
			R"(
				#define W_WIDTH (2*RADIUS_X+1)
				#define W_HEIGHT (2*RADIUS_Y+1)

				#define X_LOCAL_WIDTH (WORK_GROUP_SIZE_2D+2*RADIUS_X)
				#define X_LOCAL_HEIGHT (WORK_GROUP_SIZE_2D+2*RADIUS_Y)

				inline DATATYPE readX(read_only __global DATATYPE* X, int x, int y,
					int X_width, int X_height, int X_offset)
				{
					if (x>=0 && y>=0 && x<X_width && y<X_height)
						return X[y + x*X_height + X_offset];
					else
						return 0;
				}

				__kernel void KERNEL_NAME(
					read_only __global DATATYPE* X, int X_width, int X_height, int X_offset,
					__constant DATATYPE* W, int W_offset,
					__global DATATYPE* Y, int Y_offset)
				{
					__local DATATYPE X_local[X_LOCAL_WIDTH][X_LOCAL_HEIGHT];

					int x = get_global_id(0);
					int y = get_global_id(1);

					int xl = get_local_id(0);
					int yl = get_local_id(1);

					if (xl==WORK_GROUP_SIZE_2D-1 && yl == WORK_GROUP_SIZE_2D-1)
					{
						for (int rx=0; rx<=2*RADIUS_X; rx++)
							for (int ry=0; ry<=2*RADIUS_Y; ry++)
								X_local[xl+rx][yl+ry] = readX(X, x-RADIUS_X+rx, y-RADIUS_Y+ry, X_width, X_height, X_offset);
					}
					else if (xl==WORK_GROUP_SIZE_2D-1)
					{
						for (int rx=0; rx<=2*RADIUS_X; rx++)
							X_local[xl+rx][yl] = readX(X, x-RADIUS_X+rx, y-RADIUS_Y, X_width, X_height, X_offset);
					}
					else if (yl == WORK_GROUP_SIZE_2D-1)
					{
						for (int ry=0; ry<=2*RADIUS_Y; ry++)
							X_local[xl][yl+ry] = readX(X, x-RADIUS_X, y-RADIUS_Y+ry, X_width, X_height, X_offset);
					}
					else
						X_local[xl][yl] = readX(X, x-RADIUS_X, y-RADIUS_Y, X_width, X_height, X_offset);

					barrier(CLK_LOCAL_MEM_FENCE);

					if (x>=X_width || y>=X_height)
						return;

					DATATYPE sum = 0;
					for (int x1=0; x1<W_WIDTH; x1++)
					{
					#ifdef FLIP
						int wx = x1*W_HEIGHT+W_offset;
					#else
						int wx = (2*RADIUS_X-x1)*W_HEIGHT+W_offset;
					#endif
						int inx = x1+xl;
						for (int y1=0; y1<W_HEIGHT; y1++)
						{
							int iny = y1+yl;
							#ifdef FLIP
								sum += W[y1+wx]*X_local[inx][iny];
							#else
								sum += W[2*RADIUS_Y-y1+wx]*X_local[inx][iny];
							#endif
						}
					}
				#ifdef OVERWRITE
					Y[y+X_height*x + Y_offset] = sum;
				#else
					Y[y+X_height*x + Y_offset] += sum;
				#endif
				}
			)"
		);

		viennacl::ocl::kernel& kernel = ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_2D);
		kernel.local_work_size(1, OCL_WORK_GROUP_SIZE_2D);

		return kernel;
	}

	/** Generates the computation kernel for convolution with a arbitrary stride */
	template <class T>
	static viennacl::ocl::kernel& generate_kernel_arbitrary_stride(
		int32_t radius_x, int32_t radius_y, bool flip, bool overwrite)
	{
		std::string kernel_name =
			"convolve_arbitrary_stride_" + ocl::get_type_string<T>() + "_" +
			std::to_string(radius_x) + "_" + std::to_string(radius_y);

		if (flip) kernel_name.append("_flip");
		if (overwrite) kernel_name.append("_overwrite");

		if (ocl::kernel_exists(kernel_name))
			return ocl::get_kernel(kernel_name);

		std::string source = ocl::generate_kernel_preamble<T>(kernel_name);

		if (flip) source.append("#define FLIP\n");
		if (overwrite) source.append("#define OVERWRITE\n");

		source.append("#define RADIUS_X " + std::to_string(radius_x) + "\n");
		source.append("#define RADIUS_Y " + std::to_string(radius_y) + "\n");

		source.append(
			R"(
				#define W_WIDTH (2*RADIUS_X+1)
				#define W_HEIGHT (2*RADIUS_Y+1)

				#define X_LOCAL_WIDTH (WORK_GROUP_SIZE_2D+2*RADIUS_X)
				#define X_LOCAL_HEIGHT (WORK_GROUP_SIZE_2D+2*RADIUS_Y)

				__kernel void KERNEL_NAME(
					read_only __global DATATYPE* X, int X_width, int X_height, int X_offset,
					__constant DATATYPE* W, int W_offset,
					__global DATATYPE* Y, int Y_offset,
					int stride_x, int stride_y)
				{
					__local DATATYPE X_local[WORK_GROUP_SIZE_2D][WORK_GROUP_SIZE_2D];

					int x = get_global_id(0)*stride_x;
					int y = get_global_id(1)*stride_y;

					int Y_width = X_width/stride_x;
					int Y_height = X_height/stride_y;

					if (get_global_id(0)>=Y_width || get_global_id(1)>=Y_height)
						return;

					DATATYPE sum = 0;
					for (int x1=0; x1<W_WIDTH; x1++)
					{
					#ifdef FLIP
						int wx = x1*W_HEIGHT+W_offset;
					#else
						int wx = (2*RADIUS_X-x1)*W_HEIGHT+W_offset;
					#endif
						int inx = x1+x-RADIUS_X;
						for (int y1=0; y1<W_HEIGHT; y1++)
						{
							int iny = y1+y-RADIUS_Y;
							if (inx>=0 && iny>=0 && inx<X_width && iny<X_height)
							{
							#ifdef FLIP
								sum += W[y1+wx]*X[iny+inx*X_height+X_offset];
							#else
								sum += W[2*RADIUS_Y-y1+wx]*X[iny+inx*X_height+X_offset];
							#endif
							}
						}
					}
				#ifdef OVERWRITE
					Y[get_global_id(1)+Y_height*get_global_id(0) + Y_offset] = sum;
				#else
					Y[get_global_id(1)+Y_height*get_global_id(0) + Y_offset] += sum;
				#endif
				}
			)"
		);

		viennacl::ocl::kernel& kernel = ocl::compile_kernel(kernel_name, source);

		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_2D);
		kernel.local_work_size(1, OCL_WORK_GROUP_SIZE_2D);

		return kernel;
	}

	/** Computes the 2D convolution of X with W
	 *
	 * NOTE: For the ViennaCL backend, the size of W (number of bytes) must not exceed
	 * [CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE](http://www.khronos.org/registry/cl/sdk/1.2/docs/man/xhtml/clGetDeviceInfo.html).
	 *
	 * @param X Input image
	 * @param W Filter coefficients. The dimensions of the matrix must be odd-numbered.
	 * @param Y Output image of the same size as the input image, as the borders
	 * of the input image are implicitly padded with zeros during the computation
	 * @param flip If true the filter coefficients are flipped, performing cross-correlation
	 * instead of convolution
	 * @param overwrite If true, the values in Y are overwritten with result of the
	 * computation. Otherwise, the result is added to the existing values in Y.
	 * @param stride_x Stride in the x (column) direction
	 * @param stride_y Stride in the y (row) direction
	 */
	static void compute(CGPUMatrix<T> X, CGPUMatrix<T> W, CGPUMatrix<T> Y, bool flip ,
		bool overwrite, int32_t stride_x, int32_t stride_y, ENLAutoencoderPosition autoencoder_position = NLAP_NONE)
	{
		if (stride_x==1 && stride_y==1)
		{
			viennacl::ocl::kernel& kernel = generate_kernel_unity_stride<T>(
				(W.num_cols-1)/2, (W.num_rows-1)/2, flip, overwrite);

			kernel.global_work_size(0, ocl::align_to_multiple_2d(Y.num_cols));
			kernel.global_work_size(1, ocl::align_to_multiple_2d(Y.num_rows));

			viennacl::ocl::enqueue(kernel(
				X.vcl_matrix(), cl_int(X.num_cols), cl_int(X.num_rows), cl_int(X.offset),
				W.vcl_matrix(), cl_int(W.offset),
				Y.vcl_matrix(), cl_int(Y.offset)));
		}
		else
		{
			viennacl::ocl::kernel& kernel = generate_kernel_arbitrary_stride<T>(
				(W.num_cols-1)/2, (W.num_rows-1)/2, flip, overwrite);

			kernel.global_work_size(0, ocl::align_to_multiple_2d(Y.num_cols));
			kernel.global_work_size(1, ocl::align_to_multiple_2d(Y.num_rows));

			viennacl::ocl::enqueue(kernel(
				X.vcl_matrix(), cl_int(X.num_cols), cl_int(X.num_rows), cl_int(X.offset),
				W.vcl_matrix(), cl_int(W.offset),
				Y.vcl_matrix(), cl_int(Y.offset),
				cl_int(stride_x), cl_int(stride_y)));
		}
	}
};

#endif // HAVE_VIENNACL

}

}

}
#endif // CONVOLVE_H_
