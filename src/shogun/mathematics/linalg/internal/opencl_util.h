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

#include <shogun/mathematics/linalg/internal/opencl_config.h>
#include "Block.h"

#include <string>

namespace shogun
{

namespace linalg
{

namespace implementation
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

/** Returns true if the program has already been compiled into the current context */
inline bool program_exists(std::string prog_name)
{
	return viennacl::ocl::current_context().has_program(prog_name);
}

/** Returns a kernel in a program that has already been compiled */
inline viennacl::ocl::kernel& get_kernel(std::string prog_name, std::string kernel_name)
{
	return viennacl::ocl::current_context().get_program(prog_name).get_kernel(kernel_name);
}

/** Compiles and returns a kernel */
inline viennacl::ocl::kernel& compile_kernel(
	std::string prog_name, std::string kernel_name, std::string source)
{
	viennacl::ocl::program & prog =
		viennacl::ocl::current_context().add_program(source, prog_name);
		
	return prog.get_kernel(kernel_name);
}

}
}
}

#endif // HAVE_VIENNACL

#endif // __OPENCL_UTIL_H__
