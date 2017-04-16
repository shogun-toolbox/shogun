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

#ifndef RANDOMMATRIX_H_
#define RANDOMMATRIX_H_

#include <shogun/lib/config.h>

#ifdef HAVE_LINALG_LIB

#include <shogun/mathematics/linalg/internal/RandomVector.h>

#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUVector.h>
#include <shogun/lib/GPUMatrix.h>
#include <shogun/mathematics/linalg/internal/opencl_util.h>
#endif

namespace shogun
{

namespace linalg
{

/** @brief A generic class which is specialized for different backends for 
 * generating matrices that are filled with random numbers.
 * 
 * Supported scalar types are float32_t and float64_t
 * 
 * The default backend is the same as the one used by the Core module
 */
template <class T, Backend backend=linalg_traits<Core>::backend>
class RandomMatrix
{
public:
	/** Fills the matrix with unifromly-distributed random numbers in the 
	 * range [min_value : max_value] 
	 */
	void generate_uniform(T min_value=0, T max_value=1);
	
	/** Fills the matrix with random numbers drawn from a normal distribution 
	 * with a given mean and standard deviation
	 */
	void generate_gaussian(T mean=0, T std=1);
};

#ifdef HAVE_EIGEN3

/** @brief Specialization of RandomMatrix for the EIGEN3 backend. The matrix 
 * behaves like an SGMatrix and uses the random number generation methods of 
 * CMath to generate its numbers
 * 
 * Supported scalar types are float32_t and float64_t
 */
template<> template <class T>
class RandomMatrix<T, Backend::EIGEN3> : public SGMatrix<T>
{
public:
	using SGMatrix<T>::SGMatrix;
	
	/** Fills the matrix with unifromly-distributed random numbers in the 
	 * range [min_value : max_value] 
	 */
	void generate_uniform(T min_value=0, T max_value=1)
	{
		int32_t len = this->num_rows*this->num_cols;
		for (int32_t i=0; i<len; i++)
			this->matrix[i] = CMath::random(min_value, max_value);
	}
	
	/** Fills the matrix with random numbers drawn from a normal distribution 
	 * with mean mu and standard deviation std.
	 */
	void generate_gaussian(T mu=0, T std=1)
	{
		int32_t len = this->num_rows*this->num_cols;
		for (int32_t i=0; i<len; i++)
			this->matrix[i] = CMath::normal_random(mu, std);
	}
};

#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

/** @brief Specialization of RandomMatrix for the VIENNACL backend. The matrix 
 * behaves like a CGPUMatrix and stores its data on GPU memory. 
 * 
 * Supported scalar types are float32_t and float64_t
 * 
 * The uniform random number generation is based on the [MWC64X Random Number Generator]
 * (http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html) by 
 * David B. Thomas.
 * 
 * The Gaussian random number generation is based on the polar form of the 
 * [Box-Muller transform](http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Polar_form).
 */
template<> template <class T>
class RandomMatrix<T, Backend::VIENNACL> : public CGPUMatrix<T>
{
public:
	using CGPUMatrix<T>::CGPUMatrix;
	
	/** Fills the matrix with unifromly-distributed random numbers in the 
	 * range [min_value : max_value] 
	 */
	void generate_uniform(T min_value=0, T max_value=1)
	{
		if (states.vlen==0)
			initialize_states();
		
		int32_t len = this->num_rows*this->num_cols;
		
		viennacl::ocl::kernel& kernel = 
			RandomVector< T,Backend::VIENNACL>::generate_kernel_uniform();
		kernel.global_work_size(0, implementation::ocl::align_to_multiple_1d(len));
		
		viennacl::ocl::enqueue(kernel(
			this->vcl_matrix(), cl_int(len), cl_int(this->offset), 
			states.vcl_vector(), min_value, max_value));
	}
	
	/** Fills the matrix with random numbers drawn from a normal distribution 
	 * with mean mu and standard deviation std.
	 */
	void generate_gaussian(T mu=0, T std=1)
	{
		if (states.vlen==0)
			initialize_states();
		
		int32_t len = this->num_rows*this->num_cols;
		
		viennacl::ocl::kernel& kernel = 
			RandomVector< T,Backend::VIENNACL>::generate_kernel_gaussian();
		kernel.global_work_size(0, implementation::ocl::align_to_multiple_1d(len));
		
		viennacl::ocl::enqueue(kernel(
			this->vcl_matrix(), cl_int(len), cl_int(this->offset), 
			states.vcl_vector(), mu, std));
	}
	
protected:
	/** Initializes the states of the random number generators */
	void initialize_states()
	{
		int32_t len = this->num_rows*this->num_cols;
		SGVector<uint64_t> states_cpu(len);
		
		for (int32_t i=0; i<len; i++)
			states_cpu[i] = CMath::random();
		
		states = states_cpu;
	}
	
public:
	/** A vector of length num_rows*num_cols that contains the states of each 
	 * random number generator.
	 */
	CGPUVector<uint64_t> states;
};

#endif // HAVE_VIENNACL

}

}
#endif // HAVE_LINALG_LIB

#endif // RANDOMMATRIX_H_
