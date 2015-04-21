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

#include <shogun/lib/config.h>

#if defined(HAVE_CXX11) || defined(HAVE_CXX0X)

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/linalg.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

#ifdef HAVE_VIENNACL
#include <shogun/lib/GPUVector.h>
#include <shogun/lib/GPUMatrix.h>
#endif // HAVE_VIENNACL

#include <algorithm>
#include <gtest/gtest.h>

using namespace shogun;

TEST(Elementwise_sin, SGMatrix_NATIVE)
{
	SGMatrix<int32_t> m(3,4);
	std::iota(m.data(), m.data()+m.size(), 1);
	SGMatrix<float64_t> sin_m=linalg::elementwise_sin<linalg::Backend::NATIVE>(m);
	for (index_t i=0; i<m.size(); ++i)
		EXPECT_NEAR(CMath::sin(m[i]), sin_m[i], 1E-15);
}

TEST(Elementwise_sin, SGMatrix_NATIVE_inplace)
{
	SGMatrix<float64_t> m(3,4);
	std::iota(m.data(), m.data()+m.size(), 1);
	SGMatrix<float64_t> m_copy(m.num_rows, m.num_cols);
	std::copy(m.data(), m.data()+m.size(), m_copy.data());
	linalg::elementwise_sin_inplace<linalg::Backend::NATIVE>(m);
	for (index_t i=0; i<m.size(); ++i)
		EXPECT_NEAR(CMath::sin(m_copy[i]), m[i], 1E-15);
}

TEST(Elementwise_sin, SGVector_NATIVE)
{
	SGVector<int32_t> v(3);
	std::iota(v.data(), v.data()+v.size(), 1);
	SGVector<float64_t> sin_v=linalg::elementwise_sin<linalg::Backend::NATIVE>(v);
	for (index_t i=0; i<v.size(); ++i)
		EXPECT_NEAR(CMath::sin(v[i]), sin_v[i], 1E-15);
}

TEST(Elementwise_sin, SGVector_NATIVE_inplace)
{
	SGVector<float64_t> v(3);
	std::iota(v.data(), v.data()+v.size(), 1);
	SGVector<float64_t> v_copy(v.size());
	std::copy(v.data(), v.data()+v.size(), v_copy.data());
	linalg::elementwise_sin_inplace<linalg::Backend::NATIVE>(v);
	for (index_t i=0; i<v.size(); ++i)
		EXPECT_NEAR(CMath::sin(v_copy[i]), v[i], 1E-15);
}

TEST(Elementwise_sin, SGMatrix_NATIVE_complex128)
{
	SGMatrix<complex128_t> m(3,4);
	for (index_t i=0; i<m.size(); ++i)
		m[i]=complex128_t(i+1,i+1);
	SGMatrix<complex128_t> sin_m=linalg::elementwise_sin<linalg::Backend::NATIVE>(m);
	for (index_t i=0; i<m.size(); ++i)
	{
		EXPECT_NEAR(CMath::sin(m[i]).real(), sin_m[i].real(), 1E-15);
		EXPECT_NEAR(CMath::sin(m[i]).imag(), sin_m[i].imag(), 1E-15);
	}
}

TEST(Elementwise_sin, SGMatrix_NATIVE_complex128_inplace)
{
	SGMatrix<complex128_t> m(3,4);
	for (index_t i=0; i<m.size(); ++i)
		m[i]=complex128_t(i+1,i+1);
	SGMatrix<complex128_t> m_copy(m.num_rows, m.num_cols);
	std::copy(m.data(), m.data()+m.size(), m_copy.data());
	linalg::elementwise_sin_inplace<linalg::Backend::NATIVE>(m);
	for (index_t i=0; i<m.size(); ++i)
	{
		EXPECT_NEAR(CMath::sin(m_copy[i]).real(), m[i].real(), 1E-15);
		EXPECT_NEAR(CMath::sin(m_copy[i]).imag(), m[i].imag(), 1E-15);
	}
}

TEST(Elementwise_sin, SGVector_NATIVE_complex128)
{
	SGVector<complex128_t> v(3);
	for (index_t i=0; i<v.size(); ++i)
		v[i]=complex128_t(i+1,i+1);
	SGVector<complex128_t> sin_v=linalg::elementwise_sin<linalg::Backend::NATIVE>(v);
	for (index_t i=0; i<v.size(); ++i)
	{
		EXPECT_NEAR(CMath::sin(v[i]).real(), sin_v[i].real(), 1E-15);
		EXPECT_NEAR(CMath::sin(v[i]).imag(), sin_v[i].imag(), 1E-15);
	}
}

TEST(Elementwise_sin, SGVector_NATIVE_complex128_inplace)
{
	SGVector<complex128_t> v(3);
	for (index_t i=0; i<v.size(); ++i)
		v[i]=complex128_t(i+1,i+1);
	SGVector<complex128_t> v_copy(v.size());
	std::copy(v.data(), v.data()+v.size(), v_copy.data());
	linalg::elementwise_sin_inplace<linalg::Backend::NATIVE>(v);
	for (index_t i=0; i<v.size(); ++i)
	{
		EXPECT_NEAR(CMath::sin(v_copy[i]).real(), v[i].real(), 1E-15);
		EXPECT_NEAR(CMath::sin(v_copy[i]).imag(), v[i].imag(), 1E-15);
	}
}

#ifdef HAVE_EIGEN3
TEST(Elementwise_sin, SGMatrix_EIGEN3)
{
	SGMatrix<float64_t> m(3,4);
	std::iota(m.data(), m.data()+m.size(), 1);
	SGMatrix<float64_t> sin_m=linalg::elementwise_sin<linalg::Backend::EIGEN3>(m);
	for (index_t i=0; i<m.size(); ++i)
		EXPECT_NEAR(CMath::sin(m[i]), sin_m[i], 1E-15);
}

TEST(Elementwise_sin, SGVector_EIGEN3)
{
	SGVector<float64_t> v(3);
	std::iota(v.data(), v.data()+v.size(), 1);
	SGVector<float64_t> sin_v=linalg::elementwise_sin<linalg::Backend::EIGEN3>(v);
	for (index_t i=0; i<v.size(); ++i)
		EXPECT_NEAR(CMath::sin(v[i]), sin_v[i], 1E-15);
}
#endif // HAVE_EIGEN3

#ifdef HAVE_VIENNACL
TEST(Elementwise_sin, CGPUMatrix_VIENNACL)
{
	SGMatrix<float64_t> m(3,4);
	std::iota(m.data(), m.data()+m.size(), 1);
	CGPUMatrix<float64_t> m_gpu(m);
	CGPUMatrix<float64_t> sin_m=linalg::elementwise_sin<linalg::Backend::VIENNACL>(m_gpu);
	for (index_t i=0; i<m.size(); ++i)
		EXPECT_NEAR(CMath::sin(m[i]), sin_m[i], 1E-6);
}

TEST(Elementwise_sin, CGPUMatrix_VIENNACL_inplace)
{
	SGMatrix<float64_t> m(3,4);
	std::iota(m.data(), m.data()+m.size(), 1);
	CGPUMatrix<float64_t> m_gpu(m);
	linalg::elementwise_sin_inplace<linalg::Backend::VIENNACL>(m_gpu);
	for (index_t i=0; i<m.size(); ++i)
		EXPECT_NEAR(CMath::sin(m[i]), m_gpu[i], 1E-15);
}

TEST(Elementwise_sin, CGPUVector_VIENNACL)
{
	SGVector<float64_t> v(3);
	std::iota(v.data(), v.data()+v.size(), 1);
	CGPUVector<float64_t> v_gpu(v);
	CGPUVector<float64_t> sin_v=linalg::elementwise_sin<linalg::Backend::VIENNACL>(v_gpu);
	for (index_t i=0; i<v.size(); ++i)
		EXPECT_NEAR(CMath::sin(v[i]), sin_v[i], 1E-6);
}

TEST(Elementwise_sin, CGPUVector_VIENNACL_inplace)
{
	SGVector<float64_t> v(3);
	std::iota(v.data(), v.data()+v.size(), 1);
	CGPUVector<float64_t> v_gpu(v);
	linalg::elementwise_sin_inplace<linalg::Backend::VIENNACL>(v_gpu);
	for (index_t i=0; i<v.size(); ++i)
		EXPECT_NEAR(CMath::sin(v[i]), v_gpu[i], 1E-15);
}
#endif // HAVE_VIENNACL

TEST(Elementwise_custom, SGMatrix)
{
	SGMatrix<float64_t> m(2,2);
	std::iota(m.data(), m.data()+m.size(), 1);

	float64_t weights=0.6;
	float64_t std_dev=0.2;
	float64_t mean=0.01;

	SGMatrix<float64_t> result=linalg::elementwise_compute(m,
	[&weights, &std_dev, &mean](float64_t& sqr_dist)
	{
		float64_t outer_factor=-2*CMath::PI*CMath::sqrt(sqr_dist)*CMath::sq(weights);
		float64_t exp_factor=CMath::exp(-2*CMath::sq(CMath::PI)*sqr_dist*CMath::pow(std_dev, 2));
		float64_t sin_factor=CMath::sin(2*CMath::PI*CMath::sqrt(sqr_dist)*mean);
		return outer_factor*exp_factor*sin_factor;
	});

	for (index_t i=0; i<m.num_rows; ++i)
	{
		for (index_t j=0; j<m.num_cols; ++j)
		{
			float64_t sqr_dist=m(i, j);
			float64_t outer_factor=-2*CMath::PI*CMath::sqrt(sqr_dist)*CMath::sq(weights);
			float64_t exp_factor=CMath::exp(-2*CMath::sq(CMath::PI)*sqr_dist*CMath::pow(std_dev, 2));
			float64_t sin_factor=CMath::sin(2*CMath::PI*CMath::sqrt(sqr_dist)*mean);
			m(i, j)=outer_factor*exp_factor*sin_factor;
		}
	}

	for (index_t i=0; i<m.size(); ++i)
		EXPECT_NEAR(m.data()[i], result.data()[i], 1E-15);
}

TEST(Elementwise_custom, SGMatrix_inplace)
{
	SGMatrix<float64_t> m(2,2);
	std::iota(m.data(), m.data()+m.size(), 1);
	SGMatrix<float64_t> m_copy(m.num_rows, m.num_cols);
	std::copy(m.data(), m.data()+m.size(), m_copy.data());

	float64_t weights=0.6;
	float64_t std_dev=0.2;
	float64_t mean=0.01;

	linalg::elementwise_compute_inplace(m,
	[&weights, &std_dev, &mean](float64_t& sqr_dist)
	{
		float64_t outer_factor=-2*CMath::PI*CMath::sqrt(sqr_dist)*CMath::sq(weights);
		float64_t exp_factor=CMath::exp(-2*CMath::sq(CMath::PI)*sqr_dist*CMath::pow(std_dev, 2));
		float64_t sin_factor=CMath::sin(2*CMath::PI*CMath::sqrt(sqr_dist)*mean);
		return outer_factor*exp_factor*sin_factor;
	});

	for (index_t i=0; i<m_copy.num_rows; ++i)
	{
		for (index_t j=0; j<m_copy.num_cols; ++j)
		{
			float64_t sqr_dist=m_copy(i, j);
			float64_t outer_factor=-2*CMath::PI*CMath::sqrt(sqr_dist)*CMath::sq(weights);
			float64_t exp_factor=CMath::exp(-2*CMath::sq(CMath::PI)*sqr_dist*CMath::pow(std_dev, 2));
			float64_t sin_factor=CMath::sin(2*CMath::PI*CMath::sqrt(sqr_dist)*mean);
			m_copy(i, j)=outer_factor*exp_factor*sin_factor;
		}
	}

	for (index_t i=0; i<m.size(); ++i)
		EXPECT_NEAR(m.data()[i], m_copy.data()[i], 1E-15);
}

#ifdef HAVE_VIENNACL
TEST(Elementwise_custom, CGPUMatrix)
{
	SGMatrix<float32_t> m(2,2);
	std::iota(m.data(), m.data()+m.size(), 1);
	CGPUMatrix<float32_t> m_gpu(m);

	float32_t weights=0.6;
	float32_t std_dev=0.2;
	float32_t mean=0.01;

	std::string data_type=linalg::implementation::ocl::get_type_string<float32_t>();

	std::string s_weights=std::to_string(weights);
	std::string s_std_dev=std::to_string(std_dev);
	std::string s_mean=std::to_string(mean);
	std::string s_pi=std::to_string(CMath::PI);

	std::string operation;
	operation.append(data_type+" outer_factor=-2*"+s_pi+"*sqrt(element)*pow("+s_weights+", 2);\n");
	operation.append(data_type+" exp_factor=exp(-2*pow("+s_pi+",2)*element*pow("+s_std_dev+", 2));\n");
	operation.append(data_type+" sin_factor=sin(2*"+s_pi+"*sqrt(element)*"+s_mean+");\n");
	operation.append("return outer_factor*exp_factor*sin_factor;");

	CGPUMatrix<float32_t> result=linalg::elementwise_compute(m_gpu, operation);

	for (index_t i=0; i<m.num_rows; ++i)
	{
		for (index_t j=0; j<m.num_cols; ++j)
		{
			float32_t sqr_dist=m(i, j);
			float32_t outer_factor=-2*CMath::PI*CMath::sqrt(sqr_dist)*CMath::sq(weights);
			float32_t exp_factor=CMath::exp(-2*CMath::sq(CMath::PI)*sqr_dist*CMath::sq(std_dev));
			float32_t sin_factor=CMath::sin(2*CMath::PI*CMath::sqrt(sqr_dist)*mean);
			m(i, j)=outer_factor*exp_factor*sin_factor;
		}
	}

	for (index_t i=0; i<m.num_rows; ++i)
	{
		for (index_t j=0; j<m.num_cols; ++j)
			EXPECT_NEAR(m(i,j), result(i,j), 1E-6);
	}
}

TEST(Elementwise_custom, CGPUMatrix_inplace)
{
	SGMatrix<float32_t> m(2,2);
	std::iota(m.data(), m.data()+m.size(), 1);
	CGPUMatrix<float32_t> m_gpu(m);

	float32_t weights=0.6;
	float32_t std_dev=0.2;
	float32_t mean=0.01;

	std::string data_type=linalg::implementation::ocl::get_type_string<float32_t>();

	std::string s_weights=std::to_string(weights);
	std::string s_std_dev=std::to_string(std_dev);
	std::string s_mean=std::to_string(mean);
	std::string s_pi=std::to_string(CMath::PI);

	std::string operation;
	operation.append(data_type+" outer_factor=-2*"+s_pi+"*sqrt(element)*pow("+s_weights+", 2);\n");
	operation.append(data_type+" exp_factor=exp(-2*pow("+s_pi+",2)*element*pow("+s_std_dev+", 2));\n");
	operation.append(data_type+" sin_factor=sin(2*"+s_pi+"*sqrt(element)*"+s_mean+");\n");
	operation.append("return outer_factor*exp_factor*sin_factor;");

	linalg::elementwise_compute_inplace(m_gpu, operation);

	for (index_t i=0; i<m.num_rows; ++i)
	{
		for (index_t j=0; j<m.num_cols; ++j)
		{
			float32_t sqr_dist=m(i, j);
			float32_t outer_factor=-2*CMath::PI*CMath::sqrt(sqr_dist)*CMath::sq(weights);
			float32_t exp_factor=CMath::exp(-2*CMath::sq(CMath::PI)*sqr_dist*CMath::sq(std_dev));
			float32_t sin_factor=CMath::sin(2*CMath::PI*CMath::sqrt(sqr_dist)*mean);
			m(i, j)=outer_factor*exp_factor*sin_factor;
		}
	}

	for (index_t i=0; i<m.num_rows; ++i)
	{
		for (index_t j=0; j<m.num_cols; ++j)
			EXPECT_NEAR(m(i,j), m_gpu(i,j), 1E-6);
	}
}
#endif // HAVE_VIENNACL
#endif // defined(HAVE_CXX11) || defined(HAVE_CXX0X)
