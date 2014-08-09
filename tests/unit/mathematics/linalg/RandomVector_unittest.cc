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

#include <shogun/lib/config.h>

#ifdef HAVE_LINALG_LIB
#include <shogun/mathematics/linalg/linalg.h>
#include <shogun/mathematics/linalg/internal/RandomVector.h>
#include <gtest/gtest.h>

#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUVector.h>
#endif // HAVE_VIENNACL

using namespace shogun;

#ifdef HAVE_EIGEN3

/** Tests the uniform random number generator by computing the histogram of many
 * samples and comparing it to the uniform histogram
 */
TEST(RandomVector, eigen3_backend_uniform)
{
	CMath::init_random(12345);
	
	const int32_t N = 100000;
	
	linalg::RandomVector<float64_t, linalg::Backend::EIGEN3> v(N);
	
	v.generate_uniform(0,10);
	
	SGVector<float64_t> histogram(10);
	histogram.zero();
	
	for (int32_t i=0; i<N; i++)
		histogram[(int32_t)CMath::floor(v[i])] += 1.0/N;
	
	for (int32_t i=0; i<10; i++)
		EXPECT_NEAR(0.1, histogram[i], 0.01);
}

/** Tests the gaussian random number generator by computing the histogram of many
 * samples and comparing it to the gaussian histogram
 */
TEST(RandomVector, eigen3_backend_gaussian)
{
	CMath::init_random(12345);
	
	const int32_t N = 100000;
	
	linalg::RandomVector<float64_t, linalg::Backend::EIGEN3> v(N);
	
	v.generate_gaussian(7.5,2.0);
	
	SGVector<float64_t> histogram(15);
	histogram.zero();
	
	for (int32_t i=0; i<N; i++)
	{
		int32_t bin = CMath::floor(v[i]);
		bin = CMath::clamp(bin,0,14);
		histogram[bin] += 1.0/N;
	}
	float64_t ref[] = 
	{
		0.000575666203414724911,
		0.00240208431540700094,
		0.00924567323269862885,
		0.027824832626936951,
		0.0655995972867355476,
		0.120974368891058739,
		0.174654597118991506,
		0.197418460258255213,
		0.174654597118991506,
		0.120974368891058739,
		0.0655995972867355476,
		0.027824832626936951,
		0.00924567323269862885,
		0.00240208431540700094,
		0.000575666203414724911
	};
	
	for (int32_t i=0; i<15; i++)
		EXPECT_NEAR(1.0, histogram[i]/ref[i], 0.1);
}
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
/** Tests the uniform random number generator by computing the histogram of many
 * samples and comparing it to the uniform histogram
 */
TEST(RandomVector, viennacl_backend_uniform)
{
	CMath::init_random(100);
	
	const int32_t N = 100000;
	
	linalg::RandomVector<float64_t, linalg::Backend::VIENNACL> v(N);
	
	v.generate_uniform(0,10);
	
	SGVector<float64_t> v_cpu = v;
	
	SGVector<float64_t> histogram(10);
	histogram.zero();
	
	for (int32_t i=0; i<N; i++)
		histogram[(int32_t)CMath::floor(v_cpu[i])] += 1.0/N;
	
	for (int32_t i=0; i<10; i++)
		EXPECT_NEAR(0.1, histogram[i], 0.01);
}

/** Tests the gaussian random number generator by computing the histogram of many
 * samples and comparing it to the gaussian histogram
 */
TEST(RandomVector, viennacl_backend_gaussian)
{
	CMath::init_random(100);
	
	const int32_t N = 100000;
	
	linalg::RandomVector<float64_t, linalg::Backend::VIENNACL> v(N);
	
	v.generate_gaussian(7.5,2.0);
	
	SGVector<float64_t> v_cpu = v;
	
	SGVector<float64_t> histogram(15);
	histogram.zero();
	
	for (int32_t i=0; i<N; i++)
	{
		int32_t bin = CMath::floor(v_cpu[i]);
		bin = CMath::clamp(bin,0,14);
		histogram[bin] += 1.0/N;
	}
	float64_t ref[] = 
	{
		0.000575666203414724911,
		0.00240208431540700094,
		0.00924567323269862885,
		0.027824832626936951,
		0.0655995972867355476,
		0.120974368891058739,
		0.174654597118991506,
		0.197418460258255213,
		0.174654597118991506,
		0.120974368891058739,
		0.0655995972867355476,
		0.027824832626936951,
		0.00924567323269862885,
		0.00240208431540700094,
		0.000575666203414724911
	};
	
	for (int32_t i=0; i<15; i++)
		EXPECT_NEAR(1.0, histogram[i]/ref[i], 0.1);
}
#endif // HAVE_VIENNACL

#endif // HAVE_LINALG_LIB
