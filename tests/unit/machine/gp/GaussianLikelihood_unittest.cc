/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/base/Parameter.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/evaluation/GradientResult.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(GaussianLikelihood,get_predictive_log_probabilities)
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

	// Gaussian likelihood with sigma = 0.13
	CGaussianLikelihood* likelihood=new CGaussianLikelihood(0.13);

	SGVector<float64_t> lp=likelihood->get_predictive_log_probabilities(mu, s2, labels);

	// comparison of the log probabilities with result from GPML package
	EXPECT_NEAR(lp[0], -20.216529436592481, 1E-15);
	EXPECT_NEAR(lp[1], -4.105074362196638, 1E-15);
	EXPECT_NEAR(lp[2], -1.054630503782619, 1E-15);
	EXPECT_NEAR(lp[3], -0.773126383739893, 1E-15);
	EXPECT_NEAR(lp[4], -0.344377779665387, 1E-15);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(GaussianLikelihood,get_predictive_means)
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

	// Gaussian likelihood with sigma = 0.13
	CGaussianLikelihood* likelihood=new CGaussianLikelihood(0.13);

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

TEST(GaussianLikelihood,get_predictive_variances)
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

	// Gaussian likelihood with sigma = 0.13
	CGaussianLikelihood* likelihood=new CGaussianLikelihood(0.13);

	s2=likelihood->get_predictive_variances(mu, s2, labels);

	// comparison of the first moment with result from GPML package
	EXPECT_NEAR(s2[0], 0.116900000000000, 1E-15);
	EXPECT_NEAR(s2[1], 0.216900000000000, 1E-15);
	EXPECT_NEAR(s2[2], 1.016900000000000, 1E-15);
	EXPECT_NEAR(s2[3], 0.716900000000000, 1E-15);
	EXPECT_NEAR(s2[4], 0.316900000000000, 1E-15);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(GaussianLikelihood,get_log_probability_f)
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

	// Gaussian likelihood with sigma = 0.13
	CGaussianLikelihood* likelihood=new CGaussianLikelihood(0.13);

	SGVector<float64_t> lp=likelihood->get_log_probability_f(labels, func);

	// comparison of log likelihood with result from GPML package
	EXPECT_NEAR(lp[0], 0.677092919582238, 1E-15);
	EXPECT_NEAR(lp[1], 1.115906247984604, 1E-15);
	EXPECT_NEAR(lp[2], 0.779063286446143, 1E-15);
	EXPECT_NEAR(lp[3], 0.992445605972769, 1E-15);
	EXPECT_NEAR(lp[4], 0.678324614848509, 1E-15);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(GaussianLikelihood,get_log_probability_derivative_f)
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

	// Gaussian likelihood with sigma = 0.13
	CGaussianLikelihood* likelihood=new CGaussianLikelihood(0.13);

	SGVector<float64_t> dlp=likelihood->get_log_probability_derivative_f(labels, func, 1);
	SGVector<float64_t> d2lp=likelihood->get_log_probability_derivative_f(labels, func, 2);
	SGVector<float64_t> d3lp=likelihood->get_log_probability_derivative_f(labels, func, 3);

	// comparison of log likelihood derivatives with result from GPML package
	EXPECT_NEAR(dlp[0], -7.25030, 1E-5);
	EXPECT_NEAR(dlp[1], 0.79763, 1E-5);
	EXPECT_NEAR(dlp[2], -6.36391, 1E-5);
	EXPECT_NEAR(dlp[3], 3.90473, 1E-5);
	EXPECT_NEAR(dlp[4], 7.24024, 1E-5);

	EXPECT_NEAR(d2lp[0], -59.172, 1E-3);
	EXPECT_NEAR(d2lp[1], -59.172, 1E-3);
	EXPECT_NEAR(d2lp[2], -59.172, 1E-3);
	EXPECT_NEAR(d2lp[3], -59.172, 1E-3);
	EXPECT_NEAR(d2lp[4], -59.172, 1E-3);

	EXPECT_NEAR(d3lp[0], 0, 1E-5);
	EXPECT_NEAR(d3lp[1], 0, 1E-5);
	EXPECT_NEAR(d3lp[2], 0, 1E-5);
	EXPECT_NEAR(d3lp[3], 0, 1E-5);
	EXPECT_NEAR(d3lp[4], 0, 1E-5);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(GaussianLikelihood,get_first_derivative)
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

	// Gaussian likelihood with sigma = 0.13
	CGaussianLikelihood* likelihood=new CGaussianLikelihood(0.13);

	TParameter* param=likelihood->m_model_selection_parameters->get_parameter("sigma");

	SGVector<float64_t> lp_dhyp=likelihood->get_first_derivative(labels, func,
			param);

	// comparison of log likelihood derivative wrt sigma hyperparameter with
	// result from GPML package
	EXPECT_NEAR(lp_dhyp[0], -0.11162, 1E-5);
	EXPECT_NEAR(lp_dhyp[1], -0.98925, 1E-5);
	EXPECT_NEAR(lp_dhyp[2], -0.31556, 1E-5);
	EXPECT_NEAR(lp_dhyp[3], -0.74233, 1E-5);
	EXPECT_NEAR(lp_dhyp[4], -0.11408, 1E-5);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(GaussianLikelihood,get_second_derivative)
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

	// Gaussian likelihood with sigma = 0.13
	CGaussianLikelihood* likelihood=new CGaussianLikelihood(0.13);

	TParameter* param=likelihood->m_model_selection_parameters->get_parameter("sigma");

	SGVector<float64_t> dlp_dhyp=likelihood->get_second_derivative(labels, func,
			param);

	// comparison of log likelihood derivative wrt sigma hyperparameter
	// with result from GPML package
	EXPECT_NEAR(dlp_dhyp[0], 14.5006, 1E-4);
	EXPECT_NEAR(dlp_dhyp[1], -1.5953, 1E-4);
	EXPECT_NEAR(dlp_dhyp[2], 12.7278, 1E-4);
	EXPECT_NEAR(dlp_dhyp[3], -7.8095, 1E-4);
	EXPECT_NEAR(dlp_dhyp[4], -14.4805, 1E-4);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(GaussianLikelihood,get_third_derivative)
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

	// Gaussian likelihood with sigma = 0.13
	CGaussianLikelihood* likelihood=new CGaussianLikelihood(0.13);

	TParameter* param=likelihood->m_model_selection_parameters->get_parameter("sigma");

	SGVector<float64_t> d2lp_dhyp=likelihood->get_third_derivative(labels, func,
			param);

	// comparison of log likelihood derivative wrt sigma hyperparameter
	// with result from GPML package
	EXPECT_NEAR(d2lp_dhyp[0], 118.34, 1E-2);
	EXPECT_NEAR(d2lp_dhyp[1], 118.34, 1E-2);
	EXPECT_NEAR(d2lp_dhyp[2], 118.34, 1E-2);
	EXPECT_NEAR(d2lp_dhyp[3], 118.34, 1E-2);
	EXPECT_NEAR(d2lp_dhyp[4], 118.34, 1E-2);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(GaussianLikelihood,get_first_moments)
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

	// Gaussian likelihood with sigma = 0.13
	CGaussianLikelihood* likelihood=new CGaussianLikelihood(0.13);

	mu=likelihood->get_first_moments(mu, s2, labels);

	// comparison of the first moment with result from GPML package
	EXPECT_NEAR(mu[0], -3.15499435414885e-01, 1E-15);
	EXPECT_NEAR(mu[1], -1.01996837252190e-01, 1E-15);
	EXPECT_NEAR(mu[2], -8.45664765463661e-03, 1E-15);
	EXPECT_NEAR(mu[3], -4.05114381364208e-03, 1E-15);
	EXPECT_NEAR(mu[4], 2.06917008520038e-04, 1E-15);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(GaussianLikelihood,get_second_moments)
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

	// Gaussian likelihood with sigma = 0.13
	CGaussianLikelihood* likelihood=new CGaussianLikelihood(0.13);

	s2=likelihood->get_second_moments(mu, s2, labels);

	// comparison of the second moment with result from GPML package
	EXPECT_NEAR(s2[0], 0.0144568006843456, 1E-15);
	EXPECT_NEAR(s2[1], 0.0155832180728446, 1E-15);
	EXPECT_NEAR(s2[2], 0.0166191365916019, 1E-15);
	EXPECT_NEAR(s2[3], 0.0165016041288882, 1E-15);
	EXPECT_NEAR(s2[4], 0.0159987377721679, 1E-15);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

#endif /* HAVE_EIGEN3 */
