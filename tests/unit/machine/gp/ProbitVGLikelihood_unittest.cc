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
#include <shogun/machine/gp/ProbitVGLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(ProbitVGLikelihood,get_variational_expection)
{
	float64_t rel_tolerance = 1e-2;
	float64_t abs_tolerance;

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

	CProbitVGLikelihood *lik = new CProbitVGLikelihood();
	CBinaryLabels* lab = new CBinaryLabels(y);
	lik->set_variational_distribution(m, v, lab);

	SGVector<float64_t> aa= lik->get_variational_expection();

	// comparison of the result with result from the Matlab code
	
	abs_tolerance = CMath::get_abs_tolerance(-239.30797109313670034680, rel_tolerance);
	EXPECT_NEAR(aa[0],  -239.30797109313670034680,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.37920561632878163616, rel_tolerance);
	EXPECT_NEAR(aa[1],  -0.37920561632878163616,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.21907957635104727268, rel_tolerance);
	EXPECT_NEAR(aa[2],  -0.21907957635104727268,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.03266362390596767862, rel_tolerance);
	EXPECT_NEAR(aa[3],  -0.03266362390596767862,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.00244304943615738677, rel_tolerance);
	EXPECT_NEAR(aa[4],  -0.00244304943615738677,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-36.98213519473834764995, rel_tolerance);
	EXPECT_NEAR(aa[5],  -36.98213519473834764995,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-139.66134169388715235982, rel_tolerance);
	EXPECT_NEAR(aa[6],  -139.66134169388715235982,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-540.85980508591478610469, rel_tolerance);
	EXPECT_NEAR(aa[7],  -540.85980508591478610469,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-2103.06538992241530650062, rel_tolerance);
	EXPECT_NEAR(aa[8],  -2103.06538992241530650062,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-8510.25069018596332171001, rel_tolerance);
	EXPECT_NEAR(aa[9],  -8510.25069018596332171001,  abs_tolerance);

	// clean up
	SG_UNREF(lab);
	SG_UNREF(lik);
}

TEST(ProbitVGLikelihood,get_variational_first_derivative_wrt_sigma2)
{
	float64_t rel_tolerance = 1e-2;
	float64_t abs_tolerance;

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

	CProbitVGLikelihood *lik = new CProbitVGLikelihood();
	CBinaryLabels* lab = new CBinaryLabels(y);
	lik->set_variational_distribution(m, v, lab);

	TParameter* s2_param=lik->m_parameters->get_parameter("sigma2");

	SGVector<float64_t> dv = lik->get_variational_first_derivative(s2_param);

	// comparison of the result with result from the Matlab code
	
	abs_tolerance = CMath::get_abs_tolerance(-0.24382341475056740210, rel_tolerance);
	EXPECT_NEAR(dv[0],  -0.24382341475056740210,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.25605621136752204636, rel_tolerance);
	EXPECT_NEAR(dv[1],  -0.25605621136752204636,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.18557557851224953938, rel_tolerance);
	EXPECT_NEAR(dv[2],  -0.18557557851224953938,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.06366491848941514820, rel_tolerance);
	EXPECT_NEAR(dv[3],  -0.06366491848941514820,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.00564982253473721684, rel_tolerance);
	EXPECT_NEAR(dv[4],  -0.00564982253473721684,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.49136822415665609709, rel_tolerance);
	EXPECT_NEAR(dv[5],  -0.49136822415665609709,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.49756980386326027377, rel_tolerance);
	EXPECT_NEAR(dv[6],  -0.49756980386326027377,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.49941838587800341243, rel_tolerance);
	EXPECT_NEAR(dv[7],  -0.49941838587800341243,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.49986789129164882484, rel_tolerance);
	EXPECT_NEAR(dv[8],  -0.49986789129164882484,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.49996496386033301151, rel_tolerance);
	EXPECT_NEAR(dv[9],  -0.49996496386033301151,  abs_tolerance);


	// clean up
	SG_UNREF(lab);
	SG_UNREF(lik);
}

TEST(ProbitVGLikelihood,get_variational_first_derivative_wrt_mu)
{
	float64_t rel_tolerance = 1e-2;
	float64_t abs_tolerance;

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

	CProbitVGLikelihood *lik = new CProbitVGLikelihood();
	CBinaryLabels* lab = new CBinaryLabels(y);
	lik->set_variational_distribution(m, v, lab);

	TParameter* mu_param=lik->m_parameters->get_parameter("mu");

	SGVector<float64_t> dm = lik->get_variational_first_derivative(mu_param);

	// comparison of the result with result from the Matlab code

	abs_tolerance = CMath::get_abs_tolerance(12.41187893273750120215, rel_tolerance);
	EXPECT_NEAR(dm[0],  12.41187893273750120215,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.51455784583381758424, rel_tolerance);
	EXPECT_NEAR(dm[1],  0.51455784583381758424,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.32292507351938370963, rel_tolerance);
	EXPECT_NEAR(dm[2],  0.32292507351938370963,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.06989204661917143568, rel_tolerance);
	EXPECT_NEAR(dm[3],  0.06989204661917143568,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.00549351090856365146, rel_tolerance);
	EXPECT_NEAR(dm[4],  0.00549351090856365146,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-8.12920796722463023798, rel_tolerance);
	EXPECT_NEAR(dm[5],  -8.12920796722463023798,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-16.06674402311477223293, rel_tolerance);
	EXPECT_NEAR(dm[6],  -16.06674402311477223293,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-32.03295808195719018840, rel_tolerance);
	EXPECT_NEAR(dm[7],  -32.03295808195719018840,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-64.01602958043748969885, rel_tolerance);
	EXPECT_NEAR(dm[8],  -64.01602958043748969885,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-128.00815349131755738199, rel_tolerance);
	EXPECT_NEAR(dm[9],  -128.00815349131755738199,  abs_tolerance);

	// clean up
	SG_UNREF(lab);
	SG_UNREF(lik);
}

#endif /* HAVE_EIGEN3 */
