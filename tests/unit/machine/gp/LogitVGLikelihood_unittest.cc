/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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
 *
 * Code adapted from 
 * http://hannes.nickisch.org/code/approxXX.tar.gz
 * and the reference paper is
 * Nickisch, Hannes, and Carl Edward Rasmussen.
 * "Approximations for Binary Gaussian Process Classification."
 * Journal of Machine Learning Research 9.10 (2008).
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/BinaryLabels.h>
#include <shogun/machine/gp/LogitVGLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LogitVGLikelihood,get_variational_expection)
{
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	const index_t dim = 10;
	SGVector<float64_t> y(dim);
	SGVector<float64_t> m(dim);
	SGVector<float64_t> v(dim);

	y[0] = 1;
	y[1] = 1;
	y[2] = 1;
	y[3] = 1;
	y[4] = 1;
	y[5] = -1;
	y[6] = -1;
	y[7] = -1;
	y[8] = -1;
	y[9] = -1;

	m[0] = 0.1;
	m[1] = 0.5;
	m[2] = 1;
	m[3] = 2;
	m[4] = 4;
	m[5] = 8;
	m[6] = 16;
	m[7] = 32;
	m[8] = 64;
	m[9] = 128;

	v[0] = 0.01;
	v[1] = 0.04;
	v[2] = 0.25;
	v[3] = 0.16;
	v[4] = 1;
	v[5] = 4;
	v[6] = 16;
	v[7] = 49;
	v[8] = 64;
	v[9] = 625;

	CLogitVGLikelihood *lik = new CLogitVGLikelihood();
	CBinaryLabels* lab = new CBinaryLabels(y);
	lik->set_variational_distribution(m, v, lab);

	SGVector<float64_t> aa= lik->get_variational_expection();

	// comparison of the result with result from the Matlab code
	abs_tolorance = CMath::get_abs_tolorance(-0.6456419984162, rel_tolorance);
	EXPECT_NEAR(aa[0],  -0.6456419984162,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.4787579659559, rel_tolorance);
	EXPECT_NEAR(aa[1],  -0.4787579659559,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.3375502879114, rel_tolorance);
	EXPECT_NEAR(aa[2],  -0.3375502879114,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.1354445239772, rel_tolorance);
	EXPECT_NEAR(aa[3],  -0.1354445239772,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.0290961808890, rel_tolorance);
	EXPECT_NEAR(aa[4],  -0.0290961808890,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-8.0023826146749, rel_tolorance);
	EXPECT_NEAR(aa[5],  -8.0023826146749,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-16.0001973829933, rel_tolorance);
	EXPECT_NEAR(aa[6],  -16.0001973829933,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-32.0000083867939, rel_tolorance);
	EXPECT_NEAR(aa[7],  -32.0000083867939,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-64.0000000000000, rel_tolorance);
	EXPECT_NEAR(aa[8],  -64.0000000000000,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-128.0000007116545, rel_tolorance);
	EXPECT_NEAR(aa[9],  -128.0000007116545,  abs_tolorance);
	
	// clean up
	SG_UNREF(lab);
	SG_UNREF(lik);
}

TEST(LogitVGLikelihood,get_variational_first_derivative_wrt_sigma2)
{
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	const index_t dim = 10;
	SGVector<float64_t> y(dim);
	SGVector<float64_t> m(dim);
	SGVector<float64_t> v(dim);

	y[0] = 1;
	y[1] = 1;
	y[2] = 1;
	y[3] = 1;
	y[4] = 1;
	y[5] = -1;
	y[6] = -1;
	y[7] = -1;
	y[8] = -1;
	y[9] = -1;

	m[0] = 0.1;
	m[1] = 0.5;
	m[2] = 1;
	m[3] = 2;
	m[4] = 4;
	m[5] = 8;
	m[6] = 16;
	m[7] = 32;
	m[8] = 64;
	m[9] = 128;

	v[0] = 0.01;
	v[1] = 0.04;
	v[2] = 0.25;
	v[3] = 0.16;
	v[4] = 1;
	v[5] = 4;
	v[6] = 16;
	v[7] = 49;
	v[8] = 64;
	v[9] = 625;

	CLogitVGLikelihood *lik = new CLogitVGLikelihood();
	CBinaryLabels* lab = new CBinaryLabels(y);
	lik->set_variational_distribution(m, v, lab);

	TParameter* s2_param=lik->m_gradient_parameters->get_parameter("sigma2");

	SGVector<float64_t> dV = lik->get_variational_first_derivative(s2_param);

	// comparison of the result with result from the Matlab code
	abs_tolorance = CMath::get_abs_tolorance(-0.124380152980080, rel_tolorance);
	EXPECT_NEAR(dV[0],  -0.124380152980080,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.116551656224768, rel_tolorance);
	EXPECT_NEAR(dV[1],  -0.116551656224768,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.095970586105266, rel_tolorance);
	EXPECT_NEAR(dV[2],  -0.095970586105266,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.053914839661824, rel_tolorance);
	EXPECT_NEAR(dV[3],  -0.053914839661824,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.013148760269259, rel_tolorance);
	EXPECT_NEAR(dV[4],  -0.013148760269259,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.001097043056034, rel_tolorance);
	EXPECT_NEAR(dV[5],  -0.001097043056034,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.000068768807072, rel_tolorance);
	EXPECT_NEAR(dV[6],  -0.000068768807072,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.000001745695038, rel_tolorance);
	EXPECT_NEAR(dV[7],  -0.000001745695038,  abs_tolorance);
	EXPECT_NEAR(dV[8], -0.00000000000000303924, 1e-15);
	abs_tolorance = CMath::get_abs_tolorance(-0.000000017225282, rel_tolorance);
	EXPECT_NEAR(dV[9],  -0.000000017225282,  abs_tolorance);

	// clean up
	SG_UNREF(lab);
	SG_UNREF(lik);
}

TEST(LogitVGLikelihood,get_variational_first_derivative_wrt_mu)
{
	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	const index_t dim = 10;
	SGVector<float64_t> y(dim);
	SGVector<float64_t> m(dim);
	SGVector<float64_t> v(dim);

	y[0] = 1;
	y[1] = 1;
	y[2] = 1;
	y[3] = 1;
	y[4] = 1;
	y[5] = -1;
	y[6] = -1;
	y[7] = -1;
	y[8] = -1;
	y[9] = -1;

	m[0] = 0.1;
	m[1] = 0.5;
	m[2] = 1;
	m[3] = 2;
	m[4] = 4;
	m[5] = 8;
	m[6] = 16;
	m[7] = 32;
	m[8] = 64;
	m[9] = 128;

	v[0] = 0.01;
	v[1] = 0.04;
	v[2] = 0.25;
	v[3] = 0.16;
	v[4] = 1;
	v[5] = 4;
	v[6] = 16;
	v[7] = 49;
	v[8] = 64;
	v[9] = 625;

	CLogitVGLikelihood *lik = new CLogitVGLikelihood();
	CBinaryLabels* lab = new CBinaryLabels(y);
	lik->set_variational_distribution(m, v, lab);

	TParameter* mu_param=lik->m_gradient_parameters->get_parameter("mu");

	SGVector<float64_t> dm = lik->get_variational_first_derivative(mu_param);

	// comparison of the result with result from the Matlab code
	abs_tolorance = CMath::get_abs_tolorance(0.475082796502344, rel_tolorance);
	EXPECT_NEAR(dm[0],  0.475082796502344,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.378671358541127, rel_tolorance);
	EXPECT_NEAR(dm[1],  0.378671358541127,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.279419184756701, rel_tolorance);
	EXPECT_NEAR(dm[2],  0.279419184756701,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.125525238882835, rel_tolorance);
	EXPECT_NEAR(dm[3],  0.125525238882835,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.028103984302611, rel_tolorance);
	EXPECT_NEAR(dm[4],  0.028103984302611,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.997688981567585, rel_tolorance);
	EXPECT_NEAR(dm[5],  -0.997688981567585,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.999831811177761, rel_tolorance);
	EXPECT_NEAR(dm[6],  -0.999831811177761,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.999994480524884, rel_tolorance);
	EXPECT_NEAR(dm[7],  -0.999994480524884,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.999999999999995, rel_tolorance);
	EXPECT_NEAR(dm[8],  -0.999999999999995,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.999999840882068, rel_tolorance);
	EXPECT_NEAR(dm[9],  -0.999999840882068,  abs_tolorance);

	// clean up
	SG_UNREF(lab);
	SG_UNREF(lik);
}

#endif /* HAVE_EIGEN3 */
