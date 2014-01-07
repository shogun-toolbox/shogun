/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <lib/config.h>

#ifdef HAVE_EIGEN3

#include <base/Parameter.h>
#include <labels/RegressionLabels.h>
#include <features/DenseFeatures.h>
#include <machine/gp/StudentsTLikelihood.h>
#include <evaluation/GradientResult.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(StudentsTLikelihood,get_predictive_log_probabilities)
{
	// create some easy data:
	// mu(x) approximately equals to (x^3+sin(x)^2)/10, y=0
	index_t n=5;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> s2(n);
	SGVector<float64_t> mu(n);

	lab.set_const(0.0);

	s2[0]=0.1;
	s2[1]=0.2;
	s2[2]=1.0;
	s2[3]=0.7;
	s2[4]=0.3;

	mu[0]=-2.18236;
	mu[1]=-1.30906;
	mu[2]=-0.50885;
	mu[3]=-0.17185;
	mu[4]=0.00388;

	// shogun representation of labels
	CRegressionLabels* labels=new CRegressionLabels(lab);

	// Stundent's-t likelihood with sigma = 0.17, df = 3
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(0.17, 3);

	SGVector<float64_t> lp=likelihood->get_predictive_log_probabilities(mu, s2, labels);

	// comparison of the log probability moment with result from GPstuff package
	EXPECT_NEAR(lp[0], -7.048027785472147, 1E-10);
	EXPECT_NEAR(lp[1], -3.450707321209273, 1E-10);
	EXPECT_NEAR(lp[2], -1.073679429165726, 1E-10);
	EXPECT_NEAR(lp[3], -0.804586438080387, 1E-10);
	EXPECT_NEAR(lp[4], -0.406520554213667, 1E-10);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(StudentsTLikelihood,get_predictive_means)
{
	// create some easy data:
	// mu(x) approximately equals to (x^3+sin(x)^2)/10, y=0
	index_t n=5;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> s2(n);
	SGVector<float64_t> mu(n);

	lab.set_const(0.0);

	s2[0]=0.1;
	s2[1]=0.2;
	s2[2]=1.0;
	s2[3]=0.7;
	s2[4]=0.3;

	mu[0]=-2.18236;
	mu[1]=-1.30906;
	mu[2]=-0.50885;
	mu[3]=-0.17185;
	mu[4]=0.00388;

	// shogun representation of labels
	CRegressionLabels* labels=new CRegressionLabels(lab);

	// Stundent's-t likelihood with sigma = 0.17, df = 3
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(0.17, 3);

	mu=likelihood->get_predictive_means(mu, s2, labels);

	// comparison of the first moment with result from GPML package
	EXPECT_NEAR(mu[0], -2.18236000000000008, 1E-15);
	EXPECT_NEAR(mu[1], -1.30905999999999989, 1E-15);
	EXPECT_NEAR(mu[2], -0.50885000000000002, 1E-15);
	EXPECT_NEAR(mu[3], -0.17185000000000000, 1E-15);
	EXPECT_NEAR(mu[4], 0.00388000000000000, 1E-15);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(StudentsTLikelihood,get_predictive_variances)
{
	// create some easy data:
	// mu(x) approximately equals to (x^3+sin(x)^2)/10, y=0
	index_t n=5;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> s2(n);
	SGVector<float64_t> mu(n);

	lab.set_const(0.0);

	s2[0]=0.1;
	s2[1]=0.2;
	s2[2]=1.0;
	s2[3]=0.7;
	s2[4]=0.3;

	mu[0]=-2.18236;
	mu[1]=-1.30906;
	mu[2]=-0.50885;
	mu[3]=-0.17185;
	mu[4]=0.00388;

	// shogun representation of labels
	CRegressionLabels* labels=new CRegressionLabels(lab);

	// Stundent's-t likelihood with sigma = 0.17, df = 3
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(0.17, 3);

	s2=likelihood->get_predictive_variances(mu, s2, labels);

	// comparison of the first moment with result from GPML package
	EXPECT_NEAR(s2[0], 0.186700000000000, 1E-15);
	EXPECT_NEAR(s2[1], 0.286700000000000, 1E-15);
	EXPECT_NEAR(s2[2], 1.086700000000000, 1E-15);
	EXPECT_NEAR(s2[3], 0.786700000000000, 1E-15);
	EXPECT_NEAR(s2[4], 0.386700000000000, 1E-15);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(StudentsTLikelihood,get_log_probability_f)
{
	// create some easy data:
	// f(x) approximately equals to (x^3 + sin(x)^2)/10, y = f(x) + noise
	index_t n=5;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> func(n);

	lab[0]=-2.30489;
	lab[1]=-1.29558;
	lab[2]=-0.61640;
	lab[3]=-0.10586;
	lab[4]=0.12624;

	func[0]=-2.18236;
	func[1]=-1.30906;
	func[2]=-0.50885;
	func[3]=-0.17185;
	func[4]=0.00388;

	// shogun representation of labels
	CRegressionLabels* labels=new CRegressionLabels(lab);

	// Stundent's-t likelihood with sigma = 0.17, df = 3
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(0.17, 3);

	SGVector<float64_t> lp=likelihood->get_log_probability_f(labels, func);

	// comparison of log likelihood with result from GPML package
	EXPECT_NEAR(lp[0], 0.451653700662802, 1E-15);
	EXPECT_NEAR(lp[1], 0.766880674048690, 1E-15);
	EXPECT_NEAR(lp[2], 0.520599181456425, 1E-15);
	EXPECT_NEAR(lp[3], 0.673055347320566, 1E-15);
	EXPECT_NEAR(lp[4], 0.452472466723325, 1E-15);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(StudentsTLikelihood,get_log_probability_derivative_f)
{
	// create some easy data:
	// f(x) approximately equals to (x^3 + sin(x)^2)/10, y = f(x) + noise
	index_t n=5;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> func(n);

	lab[0]=-2.30489;
	lab[1]=-1.29558;
	lab[2]=-0.61640;
	lab[3]=-0.10586;
	lab[4]=0.12624;

	func[0]=-2.18236;
	func[1]=-1.30906;
	func[2]=-0.50885;
	func[3]=-0.17185;
	func[4]=0.00388;

	// shogun representation of labels
	CRegressionLabels* labels=new CRegressionLabels(lab);

	// Stundent's-t likelihood with sigma = 0.17, df = 3
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(0.17, 3);

	SGVector<float64_t> dlp=likelihood->get_log_probability_derivative_f(labels, func, 1);
	SGVector<float64_t> d2lp=likelihood->get_log_probability_derivative_f(labels, func, 2);
	SGVector<float64_t> d3lp=likelihood->get_log_probability_derivative_f(labels, func, 3);

	// comparison of log likelihood derivatives with result from GPML package
	EXPECT_NEAR(dlp[0], -4.81863, 1E-5);
	EXPECT_NEAR(dlp[1], 0.62061, 1E-5);
	EXPECT_NEAR(dlp[2], -4.37787, 1E-5);
	EXPECT_NEAR(dlp[3], 2.89892, 1E-5);
	EXPECT_NEAR(dlp[4], 4.81391, 1E-5);

	EXPECT_NEAR(d2lp[0], -27.717, 1E-3);
	EXPECT_NEAR(d2lp[1], -45.847, 1E-3);
	EXPECT_NEAR(d2lp[2], -31.123, 1E-3);
	EXPECT_NEAR(d2lp[3], -39.728, 1E-3);
	EXPECT_NEAR(d2lp[4], -27.755, 1E-3);

	EXPECT_NEAR(d3lp[0], 228.305, 1E-3);
	EXPECT_NEAR(d3lp[1], -42.740, 1E-3);
	EXPECT_NEAR(d3lp[2], 225.352, 1E-3);
	EXPECT_NEAR(d3lp[3], -178.842, 1E-3);
	EXPECT_NEAR(d3lp[4], -228.307, 1E-3);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(StudentsTLikelihood,get_first_derivative)
{
	// create some easy data:
	// f(x) approximately equals to (x^3 + sin(x)^2)/10, y = f(x) + noise
	index_t n=5;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> func(n);

	lab[0]=-2.30489;
	lab[1]=-1.29558;
	lab[2]=-0.61640;
	lab[3]=-0.10586;
	lab[4]=0.12624;

	func[0]=-2.18236;
	func[1]=-1.30906;
	func[2]=-0.50885;
	func[3]=-0.17185;
	func[4]=0.00388;

	// shogun representation of labels
	CRegressionLabels* labels=new CRegressionLabels(lab);

	// Stundent's-t likelihood with sigma = 0.17, df = 3
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(0.17, 3);

	TParameter* param1=likelihood->m_model_selection_parameters->get_parameter("sigma");
	TParameter* param2=likelihood->m_model_selection_parameters->get_parameter("df");

	SGVector<float64_t> lp_dhyp1=likelihood->get_first_derivative(labels, func,
			param1);
	SGVector<float64_t> lp_dhyp2=likelihood->get_first_derivative(labels, func,
			param2);

	// comparison of log likelihood derivative wrt sigma and df hyperparameter
	// with result from GPML package
	EXPECT_NEAR(lp_dhyp1[0], -0.40957, 1E-5);
	EXPECT_NEAR(lp_dhyp1[1], -0.99163, 1E-5);
	EXPECT_NEAR(lp_dhyp1[2], -0.52916, 1E-5);
	EXPECT_NEAR(lp_dhyp1[3], -0.80870, 1E-5);
	EXPECT_NEAR(lp_dhyp1[4], -0.41097, 1E-5);

	EXPECT_NEAR(lp_dhyp2[0], 0.090063, 1E-6);
	EXPECT_NEAR(lp_dhyp2[1], 0.053656, 1E-6);
	EXPECT_NEAR(lp_dhyp2[2], 0.084673, 1E-6);
	EXPECT_NEAR(lp_dhyp2[3], 0.067721, 1E-6);
	EXPECT_NEAR(lp_dhyp2[4], 0.090007, 1E-6);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(StudentsTLikelihood,get_second_derivative)
{
	// create some easy data:
	// f(x) approximately equals to (x^3 + sin(x)^2)/10, y = f(x) + noise
	index_t n=5;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> func(n);

	lab[0]=-2.30489;
	lab[1]=-1.29558;
	lab[2]=-0.61640;
	lab[3]=-0.10586;
	lab[4]=0.12624;

	func[0]=-2.18236;
	func[1]=-1.30906;
	func[2]=-0.50885;
	func[3]=-0.17185;
	func[4]=0.00388;

	// shogun representation of labels
	CRegressionLabels* labels=new CRegressionLabels(lab);

	// Stundent's-t likelihood with sigma = 0.17, df = 3
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(0.17, 3);

	TParameter* param1=likelihood->m_model_selection_parameters->get_parameter("sigma");
	TParameter* param2=likelihood->m_model_selection_parameters->get_parameter("df");

	SGVector<float64_t> dlp_dhyp1=likelihood->get_second_derivative(labels,
			func, param1);
	SGVector<float64_t> dlp_dhyp2=likelihood->get_second_derivative(labels,
			func, param2);

	// comparison of log likelihood derivative wrt sigma and df hyperparameter
	// with result from GPML package
	EXPECT_NEAR(dlp_dhyp1[0], 8.2147, 1E-4);
	EXPECT_NEAR(dlp_dhyp1[1], -1.2386, 1E-4);
	EXPECT_NEAR(dlp_dhyp1[2], 7.7251, 1E-4);
	EXPECT_NEAR(dlp_dhyp1[3], -5.5206, 1E-4);
	EXPECT_NEAR(dlp_dhyp1[4], -8.2101, 1E-4);

	EXPECT_NEAR(dlp_dhyp2[0], 0.32893, 1E-5);
	EXPECT_NEAR(dlp_dhyp2[1], -0.10257, 1E-5);
	EXPECT_NEAR(dlp_dhyp2[2], 0.38610, 1E-5);
	EXPECT_NEAR(dlp_dhyp2[3], -0.39073, 1E-5);
	EXPECT_NEAR(dlp_dhyp2[4], -0.32973, 1E-5);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(StudentsTLikelihood,get_third_derivative)
{
	// create some easy data:
	// f(x) approximately equals to (x^3 + sin(x)^2)/10, y = f(x) + noise
	index_t n=5;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> func(n);

	lab[0]=-2.30489;
	lab[1]=-1.29558;
	lab[2]=-0.61640;
	lab[3]=-0.10586;
	lab[4]=0.12624;

	func[0]=-2.18236;
	func[1]=-1.30906;
	func[2]=-0.50885;
	func[3]=-0.17185;
	func[4]=0.00388;

	// shogun representation of labels
	CRegressionLabels* labels=new CRegressionLabels(lab);

	// Stundent's-t likelihood with sigma = 0.17, df = 3
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(0.17, 3);

	TParameter* param1=likelihood->m_model_selection_parameters->get_parameter("sigma");
	TParameter* param2=likelihood->m_model_selection_parameters->get_parameter("df");

	SGVector<float64_t> d2lp_dhyp1=likelihood->get_third_derivative(labels,
			func, param1);
	SGVector<float64_t> d2lp_dhyp2=likelihood->get_third_derivative(labels,
			func, param2);

	// comparison of log likelihood derivative wrt sigma and df hyperparameter
	// with result from GPML package
	EXPECT_NEAR(d2lp_dhyp1[0], 27.459, 1E-3);
	EXPECT_NEAR(d2lp_dhyp1[1], 91.118, 1E-3);
	EXPECT_NEAR(d2lp_dhyp1[2], 38.009, 1E-3);
	EXPECT_NEAR(d2lp_dhyp1[3], 67.654, 1E-3);
	EXPECT_NEAR(d2lp_dhyp1[4], 27.575, 1E-3);

	EXPECT_NEAR(d2lp_dhyp2[0], -4.7053, 1E-4);
	EXPECT_NEAR(d2lp_dhyp2[1], 7.4491, 1E-4);
	EXPECT_NEAR(d2lp_dhyp2[2], -2.8918, 1E-4);
	EXPECT_NEAR(d2lp_dhyp2[3], 2.6874, 1E-4);
	EXPECT_NEAR(d2lp_dhyp2[4], -4.6860, 1E-4);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(StudentsTLikelihood,get_first_moments)
{
	// create some easy data:
	// mu(x) approximately equals to (x^3+sin(x)^2)/10, y=0
	index_t n=5;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> s2(n);
	SGVector<float64_t> mu(n);

	lab.set_const(0.0);

	s2[0]=0.1;
	s2[1]=0.2;
	s2[2]=1.0;
	s2[3]=0.7;
	s2[4]=0.3;

	mu[0]=-2.18236;
	mu[1]=-1.30906;
	mu[2]=-0.50885;
	mu[3]=-0.17185;
	mu[4]=0.00388;

	// shogun representation of labels
	CRegressionLabels* labels=new CRegressionLabels(lab);

	// Stundent's-t likelihood with sigma = 0.13, df = 4
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(0.13, 4);

	mu=likelihood->get_first_moments(mu, s2, labels);

	// comparison of the first moment with result from GPstuff package
	EXPECT_NEAR(mu[0], -1.91822633104012e+00, 1E-10);
	EXPECT_NEAR(mu[1], -2.38637983911711e-01, 1E-10);
	EXPECT_NEAR(mu[2], -1.51721564537775e-02, 1E-10);
	EXPECT_NEAR(mu[3], -7.02852818764603e-03, 1E-10);
	EXPECT_NEAR(mu[4], 3.26984704842959e-04, 1E-10);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(StudentsTLikelihood,get_second_moments)
{
	// create some easy data:
	// mu(x) approximately equals to (x^3+sin(x)^2)/10, y=0
	index_t n=5;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> s2(n);
	SGVector<float64_t> mu(n);

	lab.set_const(0.0);

	s2[0]=0.1;
	s2[1]=0.2;
	s2[2]=1.0;
	s2[3]=0.7;
	s2[4]=0.3;

	mu[0]=-2.18236;
	mu[1]=-1.30906;
	mu[2]=-0.50885;
	mu[3]=-0.17185;
	mu[4]=0.00388;

	// shogun representation of labels
	CRegressionLabels* labels=new CRegressionLabels(lab);

	// Stundent's-t likelihood with sigma = 0.13, df = 4
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(0.13, 4);

	s2=likelihood->get_second_moments(mu, s2, labels);

	// comparison of the second moment with result from GPstuff package
	EXPECT_NEAR(s2[0], 0.1166949785176628, 1E-10);
	EXPECT_NEAR(s2[1], 0.0821809540192921, 1E-10);
	EXPECT_NEAR(s2[2], 0.0301390579119477, 1E-10);
	EXPECT_NEAR(s2[3], 0.0286878038416459, 1E-10);
	EXPECT_NEAR(s2[4], 0.0252824052815258, 1E-10);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

#endif /* HAVE_EIGEN3 */
