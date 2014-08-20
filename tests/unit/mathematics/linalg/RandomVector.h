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

#ifndef RANDOMVECTOR_H_
#define RANDOMVECTOR_H_

#include <shogun/lib/config.h>

#ifdef HAVE_LINALG_LIB

#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUVector.h>
#include <shogun/mathematics/linalg/internal/opencl_util.h>
#endif

namespace shogun
{

namespace linalg
{

/** @brief A generic class which is specialized for different backends for 
 * generating vectors that are filled with random numbers.
 * 
 * Supported scalar types are float32_t and float64_t
 * 
 * The default backend is the same as the one used by the Core module
 */
template <class T, Backend backend=linalg_traits<Core>::backend>
class RandomVector
{
public:
	/** Fills the vector with unifromly-distributed random numbers in the 
	 * range [min_value : max_value] 
	 */
	void generate_uniform(T min_value=0, T max_value=1);
	
	/** Fills the vector with random numbers drawn from a normal distribution 
	 * with a given mean and standard deviation
	 */
	void generate_gaussian(T mean=0, T std=1);
};

#ifdef HAVE_EIGEN3

/** @brief Specialization of RandomVector for the EIGEN3 backend. The vector 
 * behaves like an SGVector and uses the random number generation methods of 
 * CMath to generate its numbers
 * 
 * Supported scalar types are float32_t and float64_t
 */
template<> template <class T>
class RandomVector<T, Backend::EIGEN3> : public SGVector<T>
{
public:
	using SGVector<T>::SGVector;
	
	/** Fills the vector with unifromly-distributed random numbers in the 
	 * range [min_value : max_value] 
	 */
	void generate_uniform(T min_value=0, T max_value=1)
	{
		for (int32_t i=0; i<this->vlen; i++)
			this->vector[i] = CMath::random(min_value, max_value);
	}
	
	/** Fills the vector with random numbers drawn from a normal distribution 
	 * with mean mu and standard deviation std.
	 */
	void generate_gaussian(T mu=0, T std=1)
	{
		for (int32_t i=0; i<this->vlen; i++)
			this->vector[i] = CMath::normal_random(mu, std);
	}
};

#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL

/** @brief Specialization of RandomVector for the VIENNACL backend. The vector 
 * behaves like a CGPUVector and stores its data on GPU memory. 
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
class RandomVector<T, Backend::VIENNACL> : public CGPUVector<T>
{
	template <class,Backend> friend class RandomMatrix;
	
public:
	using CGPUVector<T>::CGPUVector;
	
	/** Fills the vector with unifromly-distributed random numbers in the 
	 * range [min_value : max_value] 
	 */
	void generate_uniform(T min_value=0, T max_value=1)
	{
		if (states.vlen==0)
			initialize_states();
		
		viennacl::ocl::kernel& kernel = generate_kernel_uniform();
		kernel.global_work_size(0, implementation::ocl::align_to_multiple_1d(this->vlen));
		
		viennacl::ocl::enqueue(kernel(
			this->vcl_vector(), cl_int(this->vlen), cl_int(this->offset), 
			states.vcl_vector(), min_value, max_value));
	}
	
	/** Fills the vector with random numbers drawn from a normal distribution 
	 * with mean mu and standard deviation std.
	 */
	void generate_gaussian(T mu=0, T std=1)
	{
		if (states.vlen==0)
			initialize_states();
		
		viennacl::ocl::kernel& kernel = generate_kernel_gaussian();
		kernel.global_work_size(0, implementation::ocl::align_to_multiple_1d(this->vlen));
		
		viennacl::ocl::enqueue(kernel(
			this->vcl_vector(), cl_int(this->vlen), cl_int(this->offset), 
			states.vcl_vector(), mu, std));
	}
	
protected:
	/** Initializes the states of the random number generators */
	void initialize_states()
	{
		SGVector<uint64_t> states_cpu(this->vlen);
		
		for (int32_t i=0; i<this->vlen; i++)
			states_cpu[i] = CMath::random();
		
		states = states_cpu;
	}
	
	/** Generates a kernel that draws samples from a uniform distribution. */
	static viennacl::ocl::kernel& generate_kernel_uniform()
	{
		std::string kernel_name = "generate_random_uniform_" + 
			implementation::ocl::get_type_string<T>();
		
		if (implementation::ocl::kernel_exists(kernel_name))
			return implementation::ocl::get_kernel(kernel_name);
		
		std::string source = implementation::ocl::generate_kernel_preamble<T>(kernel_name);
		
		source.append(
			R"(
				inline uint MWC64X_step(__global ulong* state)
				{
					ulong s = *state;
					uint c = s>>32, x = s&0xFFFFFFFF;
					*state = x*((ulong)4294883355U) + c;
					return x^c;
				}
				
				__kernel void KERNEL_NAME(
					__global DATATYPE* vec, int size, int offset,
					__global ulong* states, DATATYPE min_value, DATATYPE max_value)
				{
					int i = get_global_id(0);
					
					if (i>=size)
						return;
					
					uint x = MWC64X_step(states+i);
					
					DATATYPE s = (DATATYPE)(x)/0xFFFFFFFF;
					
					vec[i+offset] = s*(max_value-min_value) + min_value;
				}
			)"
		);
		
		viennacl::ocl::kernel& kernel = 
			implementation::ocl::compile_kernel(kernel_name, source);
		
		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);
		
		return kernel;
	}
	
	/** Generates a kernel that draws samples from a Gaussian distribution */
	static viennacl::ocl::kernel& generate_kernel_gaussian()
	{
		std::string kernel_name = "generate_random_gaussian_" + 
			implementation::ocl::get_type_string<T>();
		
		if (implementation::ocl::kernel_exists(kernel_name))
			return implementation::ocl::get_kernel(kernel_name);
		
		std::string source = implementation::ocl::generate_kernel_preamble<T>(kernel_name);
		
		source.append(
			R"(
				inline DATATYPE MWC64X_step(__global ulong* state)
				{
					ulong s = *state;
					uint c = s>>32, x = s&0xFFFFFFFF;
					*state = x*((ulong)4294883355U) + c;
					return (DATATYPE)(x^c)/0xFFFFFFFF;
				}
				
				__kernel void KERNEL_NAME(
					__global DATATYPE* vec, int size, int offset,
					__global ulong* states, DATATYPE mean, DATATYPE std)
				{
					int i = get_global_id(0);
					
					if (i>=size)
						return;
					
					DATATYPE u1, u2, s;
					do
					{
						u1 = MWC64X_step(states+i) * 2 - 1;
						u2 = MWC64X_step(states+i) * 2 - 1;
						
						s = u1*u1 + u2*u2;
					} while ((s == 0) || (s >= 1));

					DATATYPE sample = u1*sqrt(-2.0*log(s)/s);
					
					vec[i+offset] = sample*std + mean;
				}
			)"
		);
		
		viennacl::ocl::kernel& kernel = 
			implementation::ocl::compile_kernel(kernel_name, source);
		
		kernel.local_work_size(0, OCL_WORK_GROUP_SIZE_1D);
		
		return kernel;
	}
	
public:
	/** A vector of length vlen that contains the states of each random number 
	 * generator.
	 */
	CGPUVector<uint64_t> states;
};

#endif // HAVE_VIENNACL

}

}
#endif // HAVE_LINALG_LIB

#endif // RANDOMVECTOR_H_
