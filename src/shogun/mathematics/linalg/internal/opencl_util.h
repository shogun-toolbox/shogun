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
 * Written (W) 2014 Khaled Nasr
 */

/** Utility functions for OpenCL */
#ifndef __OPENCL_UTIL_H__
#define __OPENCL_UTIL_H__

#include <shogun/lib/config.h>
#ifdef HAVE_VIENNACL
#include <viennacl/ocl/backend.hpp>
#include <viennacl/ocl/kernel.hpp>
#include <viennacl/ocl/program.hpp>
#include <viennacl/ocl/utils.hpp>
#include <viennacl/tools/tools.hpp>

#include <shogun/mathematics/linalg/internal/opencl_config.h>

#include <string>

namespace shogun
{

namespace linalg
{

namespace implementation
{

namespace ocl
{

/** Returns a string representing type T */
template <class T>
std::string get_type_string()
{
	return viennacl::ocl::type_to_string<T>::apply();
}

/** Return a source code string with some appropriate definitions for:
 * DATATYPE, KERNEL_NAME, WORK_GROUP_SIZE_1D, WORK_GROUP_SIZE_2D
 *
 * If the datatype is float64_t, the appropriate pragma for enabling double
 * precision is also appended to the source code
 */
template <class T>
std::string generate_kernel_preamble(std::string kernel_name)
{
	std::string type_string = get_type_string<T>();

	std::string source = "";
	viennacl::ocl::append_double_precision_pragma<T>(viennacl::ocl::current_context(), source);
	source.append("#define DATATYPE " + type_string + "\n");
	source.append("#define KERNEL_NAME " + kernel_name + "\n");
	source.append("#define WORK_GROUP_SIZE_1D " + std::to_string(OCL_WORK_GROUP_SIZE_1D) + "\n");
	source.append("#define WORK_GROUP_SIZE_2D " + std::to_string(OCL_WORK_GROUP_SIZE_2D) + "\n");

	return source;
}

/** Returns true if the kernel has already been compiled into the current context */
inline bool kernel_exists(std::string kernel_name)
{
	return viennacl::ocl::current_context().has_program(kernel_name);
}

/** Returns a kernel that has already been compiled */
inline viennacl::ocl::kernel& get_kernel(std::string kernel_name)
{
	return viennacl::ocl::current_context().get_program(kernel_name).get_kernel(kernel_name);
}

/** Compiles and returns a kernel */
inline viennacl::ocl::kernel& compile_kernel(std::string kernel_name, std::string source)
{
	viennacl::ocl::program & prog =
		viennacl::ocl::current_context().add_program(source, kernel_name);

	return prog.get_kernel(kernel_name);
}

/** Aligns a given value to a multiple of OCL_WORK_GROUP_SIZE_1D */
inline uint32_t align_to_multiple_1d(uint32_t n)
{
	return viennacl::tools::align_to_multiple<uint32_t>(n, OCL_WORK_GROUP_SIZE_1D);
}

/** Aligns a given value to a multiple of OCL_WORK_GROUP_SIZE_2D */
inline uint32_t align_to_multiple_2d(uint32_t n)
{
	return viennacl::tools::align_to_multiple<uint32_t>(n, OCL_WORK_GROUP_SIZE_2D);
}

/** Generates a kernel that performs a single argument elementwise operation on
 * a vector
 *
 * The operation is specified by a string containing OpenCL code that performs
 * an operation on a variable called "element" and returns the result. For
 * example the operation string: "return element+1;" will produce a kernel that
 * adds 1 to all elements of a vector.
 *
 * The kernel will have the following arguments:
 * __global DATATYPE* vec, int size, int vec_offset, __global DATATYPE* result,
 * int result_offset
 */
template <class T>
viennacl::ocl::kernel& generate_single_arg_elementwise_kernel(
	std::string kernel_name, std::string operation)
{
	if (ocl::kernel_exists(kernel_name))
		return ocl::get_kernel(kernel_name);

	std::string source = ocl::generate_kernel_preamble<T>(kernel_name);

	source.append("inline DATATYPE operation(DATATYPE element)\n{\n");
	source.append(operation);
	source.append("\n}\n");

	source.append(
		R"(
			__kernel void KERNEL_NAME(
				__global DATATYPE* vec, int size, int vec_offset,
				__global DATATYPE* result, int result_offset)
			{
				int i = get_global_id(0);

				if (i<size)
					result[i+result_offset] = operation(vec[i+vec_offset]);
			}
		)"
	);

	viennacl::ocl::kernel& kernel = ocl::compile_kernel(kernel_name, source);

	kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);

	return kernel;
}

/** Generates a kernel that performs a two-argument elementwise operation on
 * a vector
 *
 * The operation is specified by a string containing OpenCL code that performs
 * an operation on variables called "element1" and "element2" and returns the result. For
 * example the operation string: "return element1+element2;" will produce a kernel that
 * adds adds two vectors together
 *
 * The kernel will have the following arguments:
 * __global DATATYPE* vec1, int size, int vec1_offset,
 * __global DATATYPE* vec2, int vec2_offset,
 * __global DATATYPE* result, int result_offset
 */
template <class T>
viennacl::ocl::kernel& generate_two_arg_elementwise_kernel(
	std::string kernel_name, std::string operation)
{
	if (ocl::kernel_exists(kernel_name))
		return ocl::get_kernel(kernel_name);

	std::string source = ocl::generate_kernel_preamble<T>(kernel_name);

	source.append("inline DATATYPE operation(DATATYPE element1, DATATYPE element2)\n{\n");
	source.append(operation);
	source.append("\n}\n");

	source.append(
		R"(
			__kernel void KERNEL_NAME(
				__global DATATYPE* vec1, int size, int vec1_offset,
				__global DATATYPE* vec2, int vec2_offset,
				__global DATATYPE* result, int result_offset)
			{
				int i = get_global_id(0);

				if (i<size)
					result[i+result_offset] =
						operation(vec1[i+vec1_offset], vec2[i+vec2_offset]);
			}
		)"
	);

	viennacl::ocl::kernel& kernel = ocl::compile_kernel(kernel_name, source);

	kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);

	return kernel;
}

}
}
}
}

#endif // HAVE_VIENNACL

#endif // __OPENCL_UTIL_H__
