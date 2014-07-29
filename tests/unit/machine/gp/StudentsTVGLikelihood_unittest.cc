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

#include <shogun/labels/RegressionLabels.h>
#include <shogun/machine/gp/StudentsTVGLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(StudentsTVGLikelihood,get_variational_expection)
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

	m[0] = 11;
	m[1] = 0.5;
	m[2] = 11;
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

	float64_t sigma = 0.4;
	float64_t df =3.0;
	CStudentsTVGLikelihood *lik = new CStudentsTVGLikelihood(sigma, df);
	CRegressionLabels* lab = new CRegressionLabels(y);
	lik->set_variational_distribution(m, v, lab);

	SGVector<float64_t> aa= lik->get_variational_expection();

	// comparison of the result with result from the Matlab code
	
	abs_tolorance = CMath::get_abs_tolorance(-11.89826714196972723414, rel_tolorance);
	EXPECT_NEAR(aa[0],  -11.89826714196972723414,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.96351546699513479499, rel_tolorance);
	EXPECT_NEAR(aa[1],  -0.96351546699513479499,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-10.76750729356339419951, rel_tolorance);
	EXPECT_NEAR(aa[2],  -10.76750729356339419951,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-2.29339420190493248342, rel_tolorance);
	EXPECT_NEAR(aa[3],  -2.29339420190493248342,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-5.84722175343603023379, rel_tolorance);
	EXPECT_NEAR(aa[4],  -5.84722175343603023379,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-10.24792766211736250170, rel_tolorance);
	EXPECT_NEAR(aa[5],  -10.24792766211736250170,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-12.76707092691365197368, rel_tolorance);
	EXPECT_NEAR(aa[6],  -12.76707092691365197368,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-15.44228554152834043123, rel_tolorance);
	EXPECT_NEAR(aa[7],  -15.44228554152834043123,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-18.20116058223524291293, rel_tolorance);
	EXPECT_NEAR(aa[8],  -18.20116058223524291293,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-20.91182393634328917642, rel_tolorance);
	EXPECT_NEAR(aa[9],  -20.91182393634328917642,  abs_tolorance);




	// clean up
	SG_UNREF(lab);
	SG_UNREF(lik);
}

TEST(StudentsTVGLikelihood,get_variational_first_derivative_wrt_sigma2)
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

	m[0] = 11;
	m[1] = 0.5;
	m[2] = 11;
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

	float64_t sigma = 0.4;
	float64_t df =3.0;
	CStudentsTVGLikelihood *lik = new CStudentsTVGLikelihood(sigma, df);
	CRegressionLabels* lab = new CRegressionLabels(y);
	lik->set_variational_distribution(m, v, lab);

	TParameter* s2_param=lik->m_parameters->get_parameter("sigma2");

	SGVector<float64_t> dv = lik->get_variational_first_derivative(s2_param);

	// comparison of the result with result from the Matlab code

	abs_tolorance = CMath::get_abs_tolorance(-0.00520794190328421284, rel_tolorance);
	EXPECT_NEAR(dv[0],  -0.00520794190328421284,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-1.12543394006989760925, rel_tolorance);
	EXPECT_NEAR(dv[1],  -1.12543394006989760925,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.01985892749503827617, rel_tolorance);
	EXPECT_NEAR(dv[2],  0.01985892749503827617,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.06063753443103937074, rel_tolorance);
	EXPECT_NEAR(dv[3],  0.06063753443103937074,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.19973602908182547244, rel_tolorance);
	EXPECT_NEAR(dv[4],  0.19973602908182547244,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.02885469495078318153, rel_tolorance);
	EXPECT_NEAR(dv[5],  0.02885469495078318153,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.00856125968069314409, rel_tolorance);
	EXPECT_NEAR(dv[6],  0.00856125968069314409,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.00216362320940863230, rel_tolorance);
	EXPECT_NEAR(dv[7],  0.00216362320940863230,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.00051160390311292137, rel_tolorance);
	EXPECT_NEAR(dv[8],  0.00051160390311292137,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.00013735836053601773, rel_tolorance);
	EXPECT_NEAR(dv[9],  0.00013735836053601773,  abs_tolorance);



	// clean up
	SG_UNREF(lab);
	SG_UNREF(lik);
}

TEST(StudentsTVGLikelihood,get_variational_first_derivative_wrt_mu)
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

	m[0] = 11;
	m[1] = 0.5;
	m[2] = 11;
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

	float64_t sigma = 0.4;
	float64_t df =3.0;
	CStudentsTVGLikelihood *lik = new CStudentsTVGLikelihood(sigma, df);
	CRegressionLabels* lab = new CRegressionLabels(y);
	lik->set_variational_distribution(m, v, lab);

	TParameter* mu_param=lik->m_parameters->get_parameter("mu");

	SGVector<float64_t> dm = lik->get_variational_first_derivative(mu_param);

	// comparison of the result with result from the Matlab code
	abs_tolorance = CMath::get_abs_tolorance(0.67693242326614289084, rel_tolorance);
	EXPECT_NEAR(dm[0],  0.67693242326614289084,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(2.50358213659824224706, rel_tolorance);
	EXPECT_NEAR(dm[1],  2.50358213659824224706,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.39906776935069920853, rel_tolorance);
	EXPECT_NEAR(dm[2],  -0.39906776935069920853,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-2.51287848419750314832, rel_tolorance);
	EXPECT_NEAR(dm[3],  -2.51287848419750314832,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-1.36471408519218395661, rel_tolorance);
	EXPECT_NEAR(dm[4],  -1.36471408519218395661,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.46693781663777955693, rel_tolorance);
	EXPECT_NEAR(dm[5],  -0.46693781663777955693,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.25091790461761104281, rel_tolorance);
	EXPECT_NEAR(dm[6],  -0.25091790461761104281,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.12757374404154861458, rel_tolorance);
	EXPECT_NEAR(dm[7],  -0.12757374404154861458,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.06310506928193947151, rel_tolorance);
	EXPECT_NEAR(dm[8],  -0.06310506928193947151,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.03233772132982468128, rel_tolorance);
	EXPECT_NEAR(dm[9],  -0.03233772132982468128,  abs_tolorance);


	// clean up
	SG_UNREF(lab);
	SG_UNREF(lik);
}

#endif /* HAVE_EIGEN3 */
