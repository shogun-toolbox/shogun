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

	m[0] = 1;
	m[1] = 0.5;
	m[2] = 1;
	m[3] = 2;
	m[4] = 4;
	m[5] = 8;
	m[6] = 16;
	m[7] = 32;
	m[8] = 64;
	m[9] = 128;

	v[0] = 1000;
	v[1] = 0.04;
	v[2] = 0.25;
	v[3] = 0.16;
	v[4] = 1;
	v[5] = 4;
	v[6] = 16;
	v[7] = 49;
	v[8] = 100;
	v[9] = 625;

	CLogitVGLikelihood *lik = new CLogitVGLikelihood();
	CBinaryLabels* lab = new CBinaryLabels(y);
	lik->set_variational_distribution(m, v, lab);

	SGVector<float64_t> aa= lik->get_variational_expection();

	// comparison of the result with result from the Matlab code
	
	abs_tolorance = CMath::get_abs_tolorance(-12.37985798146909388606, rel_tolorance);
	EXPECT_NEAR(aa[0],  -12.37985798146909388606,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.47875796595588515636, rel_tolorance);
	EXPECT_NEAR(aa[1],  -0.47875796595588515636,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.33755028791138003141, rel_tolorance);
	EXPECT_NEAR(aa[2],  -0.33755028791138003141,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.13544452397723591441, rel_tolorance);
	EXPECT_NEAR(aa[3],  -0.13544452397723591441,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.02909618088898882626, rel_tolorance);
	EXPECT_NEAR(aa[4],  -0.02909618088898882626,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-8.00238261467492328904, rel_tolorance);
	EXPECT_NEAR(aa[5],  -8.00238261467492328904,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-16.00019738299328153630, rel_tolorance);
	EXPECT_NEAR(aa[6],  -16.00019738299328153630,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-32.00000838679385850583, rel_tolorance);
	EXPECT_NEAR(aa[7],  -32.00000838679385850583,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-64.00000000036386893498, rel_tolorance);
	EXPECT_NEAR(aa[8],  -64.00000000036386893498,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-128.00000071165447934618, rel_tolorance);
	EXPECT_NEAR(aa[9],  -128.00000071165447934618,  abs_tolorance);

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

	m[0] = 1;
	m[1] = 0.5;
	m[2] = 1;
	m[3] = 2;
	m[4] = 4;
	m[5] = 8;
	m[6] = 16;
	m[7] = 32;
	m[8] = 64;
	m[9] = 128;

	v[0] = 1000;
	v[1] = 0.04;
	v[2] = 0.25;
	v[3] = 0.16;
	v[4] = 1;
	v[5] = 4;
	v[6] = 16;
	v[7] = 49;
	v[8] = 100;
	v[9] = 625;

	CLogitVGLikelihood *lik = new CLogitVGLikelihood();
	CBinaryLabels* lab = new CBinaryLabels(y);
	lik->set_variational_distribution(m, v, lab);

	TParameter* s2_param=lik->m_parameters->get_parameter("sigma2");

	SGVector<float64_t> dv = lik->get_variational_first_derivative(s2_param);

	// comparison of the result with result from the Matlab code
	
	abs_tolorance = CMath::get_abs_tolorance(-0.00643984624096141135, rel_tolorance);
	EXPECT_NEAR(dv[0],  -0.00643984624096141135,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.11655165622477028919, rel_tolorance);
	EXPECT_NEAR(dv[1],  -0.11655165622477028919,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.09597058610526683353, rel_tolorance);
	EXPECT_NEAR(dv[2],  -0.09597058610526683353,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.05391483966182461507, rel_tolorance);
	EXPECT_NEAR(dv[3],  -0.05391483966182461507,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.01314876026830374620, rel_tolorance);
	EXPECT_NEAR(dv[4],  -0.01314876026830374620,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.00109703575801320245, rel_tolorance);
	EXPECT_NEAR(dv[5],  -0.00109703575801320245,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.00006892159173584106, rel_tolorance);
	EXPECT_NEAR(dv[6],  -0.00006892159173584106,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.00000162370307891630, rel_tolorance);
	EXPECT_NEAR(dv[7],  -0.00000162370307891630,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.00000000006545277067, rel_tolorance);
	EXPECT_NEAR(dv[8],  -0.00000000006545277067,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.00000000689624744111, rel_tolorance);
	EXPECT_NEAR(dv[9],  -0.00000000689624744111,  abs_tolorance);
	
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

	m[0] = 1;
	m[1] = 0.5;
	m[2] = 1;
	m[3] = 2;
	m[4] = 4;
	m[5] = 8;
	m[6] = 16;
	m[7] = 32;
	m[8] = 64;
	m[9] = 128;

	v[0] = 1000;
	v[1] = 0.04;
	v[2] = 0.25;
	v[3] = 0.16;
	v[4] = 1;
	v[5] = 4;
	v[6] = 16;
	v[7] = 49;
	v[8] = 100;
	v[9] = 625;

	CLogitVGLikelihood *lik = new CLogitVGLikelihood();
	CBinaryLabels* lab = new CBinaryLabels(y);
	lik->set_variational_distribution(m, v, lab);

	TParameter* mu_param=lik->m_parameters->get_parameter("mu");

	SGVector<float64_t> dm = lik->get_variational_first_derivative(mu_param);

	// comparison of the result with result from the Matlab code
	
	abs_tolorance = CMath::get_abs_tolorance(0.49998947191953668767, rel_tolorance);
	EXPECT_NEAR(dm[0],  0.49998947191953668767,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.37867135854112776716, rel_tolorance);
	EXPECT_NEAR(dm[1],  0.37867135854112776716,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.27941918475670068922, rel_tolorance);
	EXPECT_NEAR(dm[2],  0.27941918475670068922,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.12552523888283501075, rel_tolorance);
	EXPECT_NEAR(dm[3],  0.12552523888283501075,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.02810398430153109028, rel_tolorance);
	EXPECT_NEAR(dm[4],  0.02810398430153109028,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.99768898565006569612, rel_tolorance);
	EXPECT_NEAR(dm[5],  -0.99768898565006569612,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.99983172195258274151, rel_tolorance);
	EXPECT_NEAR(dm[6],  -0.99983172195258274151,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.99999490674280078206, rel_tolorance);
	EXPECT_NEAR(dm[7],  -0.99999490674280078206,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.99999999979650666670, rel_tolorance);
	EXPECT_NEAR(dm[8],  -0.99999999979650666670,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.99999993818966348424, rel_tolorance);
	EXPECT_NEAR(dm[9],  -0.99999993818966348424,  abs_tolorance);


	// clean up
	SG_UNREF(lab);
	SG_UNREF(lik);
}

#endif /* HAVE_EIGEN3 */
