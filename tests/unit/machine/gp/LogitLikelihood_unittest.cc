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

#include <shogun/labels/BinaryLabels.h>
#include <shogun/machine/gp/LogitLikelihood.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LogitLikelihood,get_log_probability_f_sum)
{
	// create some easy data:
	// f(x) approximately equals to 3*sin(sin(x^2)*sin(sin(2*x))), y = sign(f(x))
	index_t n=10;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> func(n);

	lab[0]=1.0;
	lab[1]=1.0;
	lab[2]=-1.0;
	lab[3]=-1.0;
	lab[4]=1.0;
	lab[5]=-1.0;
	lab[6]=-1.0;
	lab[7]=1.0;
	lab[8]=-1.0;
	lab[9]=1.0;

	func[0]=0.889099;
	func[1]=0.350840;
	func[2]=-2.116356;
	func[3]=-0.184742;
	func[4]=0.182117;
	func[5]=-1.108930;
	func[6]=-0.062437;
	func[7]=0.482987;
	func[8]=-0.149445;
	func[9]=0.106952;

	// shogun representation of labels
	CBinaryLabels* labels=new CBinaryLabels(lab);

	// logit likelihood
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	float64_t lp=SGVector<float64_t>::sum(likelihood->get_log_probability_f(labels, func));

	// comparison of log likelihood with result from GPML package
	EXPECT_NEAR(lp, -4.8927, 1E-4);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(LogitLikelihood,get_log_probability_derivative_f)
{
	// create some easy data:
	// f(x) approximately equals to 3*sin(sin(x^2)*sin(sin(2*x))), y = sign(f(x))
	index_t n=10;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> func(n);

	lab[0]=1.0;
	lab[1]=1.0;
	lab[2]=-1.0;
	lab[3]=-1.0;
	lab[4]=1.0;
	lab[5]=-1.0;
	lab[6]=-1.0;
	lab[7]=1.0;
	lab[8]=-1.0;
	lab[9]=1.0;

	func[0]=0.889099;
	func[1]=0.350840;
	func[2]=-2.116356;
	func[3]=-0.184742;
	func[4]=0.182117;
	func[5]=-1.108930;
	func[6]=-0.062437;
	func[7]=0.482987;
	func[8]=-0.149445;
	func[9]=0.106952;

	// shogun representation of labels
	CBinaryLabels* labels=new CBinaryLabels(lab);

	// logit likelihood
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	SGVector<float64_t> dlp=likelihood->get_log_probability_derivative_f(labels, func, 1);
	SGVector<float64_t> d2lp=likelihood->get_log_probability_derivative_f(labels, func, 2);
	SGVector<float64_t> d3lp=likelihood->get_log_probability_derivative_f(labels, func, 3);

	// comparison of log likelihood derivatives with result from GPML package
	EXPECT_NEAR(dlp[0], 0.29130, 1E-5);
	EXPECT_NEAR(dlp[1], 0.41318, 1E-5);
	EXPECT_NEAR(dlp[2], -0.10752, 1E-5);
	EXPECT_NEAR(dlp[3], -0.45395, 1E-5);
	EXPECT_NEAR(dlp[4], 0.45460, 1E-5);
	EXPECT_NEAR(dlp[5], -0.24807, 1E-5);
	EXPECT_NEAR(dlp[6], -0.48440, 1E-5);
	EXPECT_NEAR(dlp[7], 0.38155, 1E-5);
	EXPECT_NEAR(dlp[8], -0.46271, 1E-5);
	EXPECT_NEAR(dlp[9], 0.47329, 1E-5);

	EXPECT_NEAR(d2lp[0], -0.206443, 1E-6);
	EXPECT_NEAR(d2lp[1], -0.242462, 1E-6);
	EXPECT_NEAR(d2lp[2], -0.095957, 1E-6);
	EXPECT_NEAR(d2lp[3], -0.247879, 1E-6);
	EXPECT_NEAR(d2lp[4], -0.247938, 1E-6);
	EXPECT_NEAR(d2lp[5], -0.186531, 1E-6);
	EXPECT_NEAR(d2lp[6], -0.249757, 1E-6);
	EXPECT_NEAR(d2lp[7], -0.235969, 1E-6);
	EXPECT_NEAR(d2lp[8], -0.248609, 1E-6);
	EXPECT_NEAR(d2lp[9], -0.249286, 1E-6);

	EXPECT_NEAR(d3lp[0], 0.0861709, 1E-7);
	EXPECT_NEAR(d3lp[1], 0.0421017, 1E-7);
	EXPECT_NEAR(d3lp[2], -0.0753232, 1E-7);
	EXPECT_NEAR(d3lp[3], -0.0228319, 1E-7);
	EXPECT_NEAR(d3lp[4], 0.0225147, 1E-7);
	EXPECT_NEAR(d3lp[5], -0.0939856, 1E-7);
	EXPECT_NEAR(d3lp[6], -0.0077945, 1E-7);
	EXPECT_NEAR(d3lp[7], 0.0559024, 1E-7);
	EXPECT_NEAR(d3lp[8], -0.0185422, 1E-7);
	EXPECT_NEAR(d3lp[9], 0.0133181, 1E-7);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

#endif /* HAVE_EIGEN3 */
