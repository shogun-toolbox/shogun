/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Wu Lin
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/lib/config.h>

#include <shogun/labels/BinaryLabels.h>
#include <shogun/machine/gp/ProbitLikelihood.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

TEST(ProbitLikelihood,get_predictive_log_probabilities)
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

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	SGVector<float64_t> lp=likelihood->get_predictive_log_probabilities(mu, s2);

	// comparison of the log probability with result from GPML package
	EXPECT_NEAR(lp[0], -0.221016102, 1E-9);
	EXPECT_NEAR(lp[1], -0.469014056, 1E-9);
	EXPECT_NEAR(lp[2], -2.699144260, 1E-9);
	EXPECT_NEAR(lp[3], -0.812691857, 1E-9);
	EXPECT_NEAR(lp[4], -0.573673124, 1E-9);
	EXPECT_NEAR(lp[5], -1.929766885, 1E-9);
	EXPECT_NEAR(lp[6], -0.728997040, 1E-9);
	EXPECT_NEAR(lp[7], -0.425655555, 1E-9);
	EXPECT_NEAR(lp[8], -0.788835673, 1E-9);
	EXPECT_NEAR(lp[9], -0.623599250, 1E-9);

	// clean up
	SG_UNREF(likelihood);
}

TEST(ProbitLikelihood,get_predictive_means)
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

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	mu=likelihood->get_predictive_means(mu, s2);

	// comparison of the first moment with result from GPML package
	EXPECT_NEAR(mu[0], 0.603407542, 1E-9);
	EXPECT_NEAR(mu[1], 0.251237577, 1E-9);
	EXPECT_NEAR(mu[2], -0.865473904, 1E-9);
	EXPECT_NEAR(mu[3], -0.112675636, 1E-9);
	EXPECT_NEAR(mu[4], 0.126904007, 1E-9);
	EXPECT_NEAR(mu[5], -0.709635922, 1E-9);
	EXPECT_NEAR(mu[6], -0.035214864, 1E-9);
	EXPECT_NEAR(mu[7], 0.306682687, 1E-9);
	EXPECT_NEAR(mu[8], -0.091252946, 1E-9);
	EXPECT_NEAR(mu[9], 0.072023442, 1E-9);

	// clean up
	SG_UNREF(likelihood);
}

TEST(ProbitLikelihood,get_predictive_variances)
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
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	s2=likelihood->get_predictive_variances(mu, s2);

	// comparison of the second moment with result from GPML package
	EXPECT_NEAR(s2[0], 0.635899337, 1E-9);
	EXPECT_NEAR(s2[1], 0.936879679, 1E-9);
	EXPECT_NEAR(s2[2], 0.250954920, 1E-9);
	EXPECT_NEAR(s2[3], 0.987304201, 1E-9);
	EXPECT_NEAR(s2[4], 0.983895372, 1E-9);
	EXPECT_NEAR(s2[5], 0.496416856, 1E-9);
	EXPECT_NEAR(s2[6], 0.998759913, 1E-9);
	EXPECT_NEAR(s2[7], 0.905945729, 1E-9);
	EXPECT_NEAR(s2[8], 0.991672899, 1E-9);
	EXPECT_NEAR(s2[9], 0.994812623, 1E-9);

	// clean up
	SG_UNREF(likelihood);
}

TEST(ProbitLikelihood,get_log_probability_f)
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

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	SGVector<float64_t> lp=likelihood->get_log_probability_f(labels, func);

	// comparison of log likelihood with result from GPML package
	EXPECT_NEAR(lp[0], -0.2069933435, 1E-9);
	EXPECT_NEAR(lp[1], -0.4507567536, 1E-9);
	EXPECT_NEAR(lp[2], -0.0173061621, 1E-9);
	EXPECT_NEAR(lp[3], -0.5563735264, 1E-9);
	EXPECT_NEAR(lp[4], -0.5581713758, 1E-9);
	EXPECT_NEAR(lp[5], -0.1435588605, 1E-9);
	EXPECT_NEAR(lp[6], -0.6445616379, 1E-9);
	EXPECT_NEAR(lp[7], -0.3776833450, 1E-9);
	EXPECT_NEAR(lp[8], -0.5808927376, 1E-9);
	EXPECT_NEAR(lp[9], -0.6114078142, 1E-9);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(ProbitLikelihood,get_log_probability_derivative_f)
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

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	SGVector<float64_t> dlp=likelihood->get_log_probability_derivative_f(labels, func, 1);
	SGVector<float64_t> d2lp=likelihood->get_log_probability_derivative_f(labels, func, 2);
	SGVector<float64_t> d3lp=likelihood->get_log_probability_derivative_f(labels, func, 3);

	// comparison of log likelihood derivatives with result from GPML package
	EXPECT_NEAR(dlp[0], 0.330485099, 1E-9);
	EXPECT_NEAR(dlp[1], 0.588766191, 1E-9);
	EXPECT_NEAR(dlp[2], -0.043234619, 1E-9);
	EXPECT_NEAR(dlp[3], -0.684114614, 1E-9);
	EXPECT_NEAR(dlp[4], 0.685675731, 1E-9);
	EXPECT_NEAR(dlp[5], -0.249014899, 1E-9);
	EXPECT_NEAR(dlp[6], -0.758565533, 1E-9);
	EXPECT_NEAR(dlp[7], 0.517941276, 1E-9);
	EXPECT_NEAR(dlp[8], -0.705243109, 1E-9);
	EXPECT_NEAR(dlp[9], 0.731067061, 1E-9);

	EXPECT_NEAR(d2lp[0], -0.403054372, 1E-9);
	EXPECT_NEAR(d2lp[1], -0.553208359, 1E-9);
	EXPECT_NEAR(d2lp[2], -0.093369078, 1E-9);
	EXPECT_NEAR(d2lp[3], -0.594397508, 1E-9);
	EXPECT_NEAR(d2lp[4], -0.595024415, 1E-9);
	EXPECT_NEAR(d2lp[5], -0.338148513, 1E-9);
	EXPECT_NEAR(d2lp[6], -0.622784225, 1E-9);
	EXPECT_NEAR(d2lp[7], -0.518422069, 1E-9);
	EXPECT_NEAR(d2lp[8], -0.602762899, 1E-9);
	EXPECT_NEAR(d2lp[9], -0.612648132, 1E-9);

	EXPECT_NEAR(d3lp[0], 0.294277068, 1E-9);
	EXPECT_NEAR(d3lp[1], 0.256742186, 1E-9);
	EXPECT_NEAR(d3lp[2], -0.162441143, 1E-9);
	EXPECT_NEAR(d3lp[3], -0.238967614, 1E-9);
	EXPECT_NEAR(d3lp[4], 0.238675932, 1E-9);
	EXPECT_NEAR(d3lp[5], -0.294376167, 1E-9);
	EXPECT_NEAR(d3lp[6], -0.225164541, 1E-9);
	EXPECT_NEAR(d3lp[7], 0.269474219, 1E-9);
	EXPECT_NEAR(d3lp[8], -0.235025555, 1E-9);
	EXPECT_NEAR(d3lp[9], 0.230230621, 1E-9);
	// clean up
	SG_UNREF(labels);

	lab[0]=1.0;
	lab[1]=1.0;
	lab[2]=1.0;
	lab[3]=1.0;
	lab[4]=1.0;
	lab[5]=-1.0;
	lab[6]=-1.0;
	lab[7]=-1.0;
	lab[8]=-1.0;
	lab[9]=-1.0;


	func[0]=241.9354699509236468202288961037993431091309;
	func[1]=2.0238097083359516403788802563212811946869;
	func[2]=4.8095242708398782127687809406779706478119;
	func[3]=5.0476194166719032807577605126425623893738;
	func[4]=11.6190485416797564255375618813559412956238;
	func[5]=23.2380970833595128510751237627118825912476;
	func[6]=46.4761941667190257021502475254237651824951;
	func[7]=85.3333397917583056369039695709943771362305;
	func[8]=140.1904854167975713608029764145612716674805;
	func[9]=176.1904854167975713608029764145612716674805;

	labels=new CBinaryLabels(lab);
	dlp=likelihood->get_log_probability_derivative_f(labels, func, 1);
	d2lp=likelihood->get_log_probability_derivative_f(labels, func, 2);
	d3lp=likelihood->get_log_probability_derivative_f(labels, func, 3);

	float64_t abs_tolerance, rel_tolerance;
	rel_tolerance=1e-2;

	abs_tolerance = CMath::get_abs_tolerance(0.0000000000000000000000000000000000000000, rel_tolerance);
	EXPECT_NEAR(dlp[0],  0.0000000000000000000000000000000000000000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0525961606087547994814457297252374701202, rel_tolerance);
	EXPECT_NEAR(dlp[1],  0.0525961606087547994814457297252374701202,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000037841107748951116192758468981249820, rel_tolerance);
	EXPECT_NEAR(dlp[2],  0.0000037841107748951116192758468981249820,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000011703950725430704780759402425238669, rel_tolerance);
	EXPECT_NEAR(dlp[3],  0.0000011703950725430704780759402425238669,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000000000000000000000000000019299030137, rel_tolerance);
	EXPECT_NEAR(dlp[4],  0.0000000000000000000000000000019299030137,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-23.2809719446751230975678481627255678176880, rel_tolerance);
	EXPECT_NEAR(dlp[5],  -23.2809719446751230975678481627255678176880,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-46.4976906821641904343778151087462902069092, rel_tolerance);
	EXPECT_NEAR(dlp[6],  -46.4976906821641904343778151087462902069092,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-85.3450553244290972543240059167146682739258, rel_tolerance);
	EXPECT_NEAR(dlp[7],  -85.3450553244290972543240059167146682739258,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-140.1976178427906631895893951877951622009277, rel_tolerance);
	EXPECT_NEAR(dlp[8],  -140.1976178427906631895893951877951622009277,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-176.1961607265705254121712641790509223937988, rel_tolerance);
	EXPECT_NEAR(dlp[9],  -176.1961607265705254121712641790509223937988,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(-0.0000000000000000000000000000000000000000, rel_tolerance);
	EXPECT_NEAR(d2lp[0],  -0.0000000000000000000000000000000000000000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.1092109765719768421643465217130142264068, rel_tolerance);
	EXPECT_NEAR(d2lp[1],  -0.1092109765719768421643465217130142264068,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000181997869348990967198250595648900685, rel_tolerance);
	EXPECT_NEAR(d2lp[2],  -0.0000181997869348990967198250595648900685,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000059077102631701492848541150404440003, rel_tolerance);
	EXPECT_NEAR(d2lp[3],  -0.0000059077102631701492848541150404440003,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000000000000000000000000000224236367964, rel_tolerance);
	EXPECT_NEAR(d2lp[4],  -0.0000000000000000000000000000224236367964,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.9981684434205588818400656236917711794376, rel_tolerance);
	EXPECT_NEAR(d2lp[5],  -0.9981684434205588818400656236917711794376,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.9995383259136347842144232345162890851498, rel_tolerance);
	EXPECT_NEAR(d2lp[6],  -0.9995383259136347842144232345162890851498,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.9998627839438671838223626764374785125256, rel_tolerance);
	EXPECT_NEAR(d2lp[7],  -0.9998627839438671838223626764374785125256,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.9999491336714748657144014032382983714342, rel_tolerance);
	EXPECT_NEAR(d2lp[8],  -0.9999491336714748657144014032382983714342,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.9999677929284884747573869390180334448814, rel_tolerance);
	EXPECT_NEAR(d2lp[9],  -0.9999677929284884747573869390180334448814,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.0000000000000000000000000000000000000000, rel_tolerance);
	EXPECT_NEAR(d3lp[0],  0.0000000000000000000000000000000000000000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.1799142301624994111364230775507166981697, rel_tolerance);
	EXPECT_NEAR(d3lp[1],  0.1799142301624994111364230775507166981697,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000837483439526362955255753273142715898, rel_tolerance);
	EXPECT_NEAR(d3lp[2],  0.0000837483439526362955255753273142715898,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000286494917886164182272389078232066595, rel_tolerance);
	EXPECT_NEAR(d3lp[3],  0.0000286494917886164182272389078232066595,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000000000000000000000000002586114214045, rel_tolerance);
	EXPECT_NEAR(d3lp[4],  0.0000000000000000000000000002586114214045,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0001559162402209324227442266419529914856, rel_tolerance);
	EXPECT_NEAR(d3lp[5],  -0.0001559162402209324227442266419529914856,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000198121972587728123471606522798538208, rel_tolerance);
	EXPECT_NEAR(d3lp[6],  -0.0000198121972587728123471606522798538208,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000032132095526549164787866175174713135, rel_tolerance);
	EXPECT_NEAR(d3lp[7],  -0.0000032132095526549164787866175174713135,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000007251051385992468567565083503723145, rel_tolerance);
	EXPECT_NEAR(d3lp[8],  -0.0000007251051385992468567565083503723145,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000003646392769951489754021167755126953, rel_tolerance);
	EXPECT_NEAR(d3lp[9],  -0.0000003646392769951489754021167755126953,  abs_tolerance);


	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(ProbitLikelihood,get_first_moments)
{
	// create some easy data:
	// mu(x) approximately equals to 3*sin(sin(x^2)*sin(sin(2*x)))
	index_t n=10;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> s2(n);
	SGVector<float64_t> mu(n);

	lab.set_const(1.0);

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

	// shogun representation of labels
	CBinaryLabels* labels=new CBinaryLabels(lab);

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	mu=likelihood->get_first_moments(mu, s2, labels);

	// comparison of the first moment with result from GPstuff package
	EXPECT_NEAR(mu[0], 0.922223587528548, 1E-10);
	EXPECT_NEAR(mu[1], 0.461442771893249, 1E-10);
	EXPECT_NEAR(mu[2], -0.747614837845594, 1E-10);
	EXPECT_NEAR(mu[3], 0.293196189957756, 1E-10);
	EXPECT_NEAR(mu[4], 0.366051285301334, 1E-10);
	EXPECT_NEAR(mu[5], -0.959118595310375, 1E-10);
	EXPECT_NEAR(mu[6], 0.521775976041116, 1E-10);
	EXPECT_NEAR(mu[7], 0.713621395240100, 1E-10);
	EXPECT_NEAR(mu[8], 0.318848193810383, 1E-10);
	EXPECT_NEAR(mu[9], 0.357538428547672, 1E-10);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}

TEST(ProbitLikelihood,get_second_moments)
{
	// create some easy data:
	// mu(x) approximately equals to 3*sin(sin(x^2)*sin(sin(2*x)))
	index_t n=10;

	SGVector<float64_t> lab(n);
	SGVector<float64_t> s2(n);
	SGVector<float64_t> mu(n);

	lab.set_const(1.0);

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

	// shogun representation of labels
	CBinaryLabels* labels=new CBinaryLabels(lab);

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	s2=likelihood->get_second_moments(mu, s2, labels);

	// comparison of the second moment with result from GPstuff package
	EXPECT_NEAR(s2[0], 0.0962253946422413, 1E-10);
	EXPECT_NEAR(s2[1], 0.1812997141010253, 1E-10);
	EXPECT_NEAR(s2[2], 0.5749194165104299, 1E-10);
	EXPECT_NEAR(s2[3], 0.5079319571460353, 1E-10);
	EXPECT_NEAR(s2[4], 0.2584379724823283, 1E-10);
	EXPECT_NEAR(s2[5], 0.0926593031160547, 1E-10);
	EXPECT_NEAR(s2[6], 0.6769334514177224, 1E-10);
	EXPECT_NEAR(s2[7], 0.4096766375142901, 1E-10);
	EXPECT_NEAR(s2[8], 0.5095184572451623, 1E-10);
	EXPECT_NEAR(s2[9], 0.3295490933402853, 1E-10);

	// clean up
	SG_UNREF(likelihood);
	SG_UNREF(labels);
}
