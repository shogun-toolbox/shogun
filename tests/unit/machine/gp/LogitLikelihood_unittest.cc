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

TEST(LogitLikelihood,get_predictive_log_probabilities)
{
	// create some easy data:
	// mu(x) approximately equals to 3*sin(sin(x^2)*sin(sin(2*x)))
	index_t n=10;

	SGVector<float64_t> s2(n);
	SGVector<float64_t> mu(n);

	s2[0]=0.1;
	s2[1]=0.2;
	s2[2]=1.0;
	s2[3]=0.7;
	s2[4]=0.3;
	s2[5]=0.1;
	s2[6]=1.0;
	s2[7]=0.5;
	s2[8]=0.7;
	s2[9]=0.4;

	mu[0]=0.889099;
	mu[1]=0.350840;
	mu[2]=-2.116356;
	mu[3]=-0.184742;
	mu[4]=0.182117;
	mu[5]=-1.108930;
	mu[6]=-0.062437;
	mu[7]=0.482987;
	mu[8]=-0.149445;
	mu[9]=0.106952;

	// logit likelihood
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	SGVector<float64_t> lp=likelihood->get_predictive_log_probabilities(mu, s2);

	// comparison of the first moment with result from GPML package
	EXPECT_NEAR(lp[0], -0.350067368640123, 1E-3);
	EXPECT_NEAR(lp[1], -0.539595630725218, 1E-3);
	EXPECT_NEAR(lp[2], -1.948742077218326, 1E-3);
	EXPECT_NEAR(lp[3], -0.776497165015726, 1E-3);
	EXPECT_NEAR(lp[4], -0.611648769423109, 1E-3);
	EXPECT_NEAR(lp[5], -1.376387887056822, 1E-3);
	EXPECT_NEAR(lp[6], -0.719289371775576, 1E-3);
	EXPECT_NEAR(lp[7], -0.499251131748656, 1E-3);
	EXPECT_NEAR(lp[8], -0.760072326237507, 1E-3);
	EXPECT_NEAR(lp[9], -0.645354698503709, 1E-3);

	// clean up
	SG_UNREF(likelihood);
}

TEST(LogitLikelihood,get_predictive_means)
{
	// create some easy data:
	// mu(x) approximately equals to 3*sin(sin(x^2)*sin(sin(2*x)))
	index_t n=10;

	SGVector<float64_t> s2(n);
	SGVector<float64_t> mu(n);

	s2[0]=0.1;
	s2[1]=0.2;
	s2[2]=1.0;
	s2[3]=0.7;
	s2[4]=0.3;
	s2[5]=0.1;
	s2[6]=1.0;
	s2[7]=0.5;
	s2[8]=0.7;
	s2[9]=0.4;

	mu[0]=0.889099;
	mu[1]=0.350840;
	mu[2]=-2.116356;
	mu[3]=-0.184742;
	mu[4]=0.182117;
	mu[5]=-1.108930;
	mu[6]=-0.062437;
	mu[7]=0.482987;
	mu[8]=-0.149445;
	mu[9]=0.106952;

	// logit likelihood
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	mu=likelihood->get_predictive_means(mu, s2);

	// comparison of the first moment with result from GPML package
	EXPECT_NEAR(mu[0], 0.4092812348789756, 1E-3);
	EXPECT_NEAR(mu[1], 0.1659678910250058, 1E-3);
	EXPECT_NEAR(mu[2], -0.7150936920105806, 1E-3);
	EXPECT_NEAR(mu[3], -0.0799709050153369, 1E-3);
	EXPECT_NEAR(mu[4], 0.0849114938067221, 1E-3);
	EXPECT_NEAR(mu[5], -0.4950221471933445, 1E-3);
	EXPECT_NEAR(mu[6], -0.0258034424328211, 1E-3);
	EXPECT_NEAR(mu[7], 0.2139700827638547, 1E-3);
	EXPECT_NEAR(mu[8], -0.0647347926399858, 1E-3);
	EXPECT_NEAR(mu[9], 0.0489529561764410, 1E-3);

	// clean up
	SG_UNREF(likelihood);
}

TEST(LogitLikelihood,get_predictive_variances)
{
	// create some easy data:
	// mu(x) approximately equals to 3*sin(sin(x^2)*sin(sin(2*x)))
	index_t n=10;

	SGVector<float64_t> s2(n);
	SGVector<float64_t> mu(n);

	s2[0]=0.1;
	s2[1]=0.2;
	s2[2]=1.0;
	s2[3]=0.7;
	s2[4]=0.3;
	s2[5]=0.1;
	s2[6]=1.0;
	s2[7]=0.5;
	s2[8]=0.7;
	s2[9]=0.4;

	mu[0]=0.889099;
	mu[1]=0.350840;
	mu[2]=-2.116356;
	mu[3]=-0.184742;
	mu[4]=0.182117;
	mu[5]=-1.108930;
	mu[6]=-0.062437;
	mu[7]=0.482987;
	mu[8]=-0.149445;
	mu[9]=0.106952;

	// logit likelihood
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	s2=likelihood->get_predictive_variances(mu, s2);

	// comparison of the second moment with result from GPML package
	EXPECT_NEAR(s2[0], 0.832488870775941, 1E-3);
	EXPECT_NEAR(s2[1], 0.972454659148712, 1E-3);
	EXPECT_NEAR(s2[2], 0.488641011646677, 1E-3);
	EXPECT_NEAR(s2[3], 0.993604654351028, 1E-3);
	EXPECT_NEAR(s2[4], 0.992790038219511, 1E-3);
	EXPECT_NEAR(s2[5], 0.754953073788091, 1E-3);
	EXPECT_NEAR(s2[6], 0.999334182358616, 1E-3);
	EXPECT_NEAR(s2[7], 0.954216803682029, 1E-3);
	EXPECT_NEAR(s2[8], 0.995809406621858, 1E-3);
	EXPECT_NEAR(s2[9], 0.997603608081587, 1E-3);

	// clean up
	SG_UNREF(likelihood);
}

TEST(LogitLikelihood,get_log_probability_f)
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

	SGVector<float64_t> lp=likelihood->get_log_probability_f(labels, func);

	// comparison of log likelihood with result from GPML package
	EXPECT_NEAR(lp[0], -0.344317042879852, 1E-15);
	EXPECT_NEAR(lp[1], -0.533034999733996, 1E-15);
	EXPECT_NEAR(lp[2], -0.113748080981196, 1E-15);
	EXPECT_NEAR(lp[3], -0.605036328325528, 1E-15);
	EXPECT_NEAR(lp[4], -0.606228789118445, 1E-15);
	EXPECT_NEAR(lp[5], -0.285112607617685, 1E-15);
	EXPECT_NEAR(lp[6], -0.662415898798726, 1E-15);
	EXPECT_NEAR(lp[7], -0.480534140470494, 1E-15);
	EXPECT_NEAR(lp[8], -0.621213812513251, 1E-15);
	EXPECT_NEAR(lp[9], -0.641100340885144, 1E-15);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(LogitLikelihood,get_log_probability_f_sum_multiple)
{
	// create some easy data:
	// f(x) approximately equals to 3*sin(sin(x^2)*sin(sin(2*x))), y = sign(f(x))
	index_t n=10;

	SGVector<float64_t> lab(n);
	SGMatrix<float64_t> func(n,2);

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

	func(0,0)=0.889099;
	func(1,0)=0.350840;
	func(2,0)=-2.116356;
	func(3,0)=-0.184742;
	func(4,0)=0.182117;
	func(5,0)=-1.108930;
	func(6,0)=-0.062437;
	func(7,0)=0.482987;
	func(8,0)=-0.149445;
	func(9,0)=0.106952;

	func(0,1)=0.889099;
	func(1,1)=0.350840;
	func(2,1)=-2.116356;
	func(3,1)=-0.184742;
	func(4,1)=0.182117;
	func(5,1)=-1.108930;
	func(6,1)=-0.062437;
	func(7,1)=0.482987;
	func(8,1)=-0.149445;
	func(9,1)=0.106952;

	// shogun representation of labels
	CBinaryLabels* labels=new CBinaryLabels(lab);

	// logit likelihood
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	SGVector<float64_t> lp=((CLikelihoodModel*)likelihood)->get_log_probability_fmatrix(labels, func);

	// comparison of log likelihood with result from GPML package
	EXPECT_NEAR(lp[0], -4.8927, 1E-4);
	EXPECT_NEAR(lp[1], -4.8927, 1E-4);

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
