/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Parijat Mazumdar
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
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

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/SoftMaxLikelihood.h>
#include <gtest/gtest.h>

using namespace shogun;

/* Results compare with softmax likelihood implementation in
 * GPStuff toolbox
 * http://becs.aalto.fi/en/research/bayes/gpstuff/install.html
 */

TEST(SoftMaxLikelihood,get_log_probabilities_f)
{
	SGMatrix<float64_t> data(7,3);

	data(0,0)=-0.8095;
	data(1,0)=-2.9443;
	data(2,0)=1.4384;
	data(3,0)=0.3252;
	data(4,0)=-0.7549;
	data(5,0)=1.3703;
	data(6,0)=-1.7115;

	data(0,1)=-0.1022;
	data(1,1)=-0.2414;
	data(2,1)=0.3192;
	data(3,1)=0.3129;
	data(4,1)=-0.8649;
	data(5,1)=-0.0301;
	data(6,1)=-0.1649;

	data(0,2)=0.6277;
	data(1,2)=1.0933;
	data(2,2)=1.1093;
	data(3,2)=-0.8637;
	data(4,2)=0.0774;
	data(5,2)=-1.2141;
	data(6,2)=-1.1135;

	SGVector<int32_t> lab(7);
	lab[0]=2;
	lab[1]=1;
	lab[2]=1;
	lab[3]=0;
	lab[4]=2;
	lab[5]=0;
	lab[6]=0;

	CMulticlassLabels* labels=new CMulticlassLabels();
	labels->set_int_labels(lab);

	CSoftMaxLikelihood* sml=new CSoftMaxLikelihood();
	SGVector<float64_t> data_vector=SGVector<float64_t>(data.matrix,data.num_rows*data.num_cols,false);
	SGVector<float64_t> v=sml->get_log_probability_f(labels,data_vector);

	float64_t ep=0.0001;
	EXPECT_NEAR(v[0],-0.5420,ep);
	EXPECT_NEAR(v[1],-1.5823,ep);
	EXPECT_NEAR(v[2],-1.8351,ep);
	EXPECT_NEAR(v[3],-0.8296,ep);
	EXPECT_NEAR(v[4],-0.6015,ep);
	EXPECT_NEAR(v[5],-0.2791,ep);
	EXPECT_NEAR(v[6],-2.0168,ep);

	SG_UNREF(sml);
	SG_UNREF(labels);
}

TEST(SoftMaxLikelihood,get_log_probability_derivative_first)
{
	SGMatrix<float64_t> data(7,3);

	data(0,0)=-0.8095;
	data(1,0)=-2.9443;
	data(2,0)=1.4384;
	data(3,0)=0.3252;
	data(4,0)=-0.7549;
	data(5,0)=1.3703;
	data(6,0)=-1.7115;

	data(0,1)=-0.1022;
	data(1,1)=-0.2414;
	data(2,1)=0.3192;
	data(3,1)=0.3129;
	data(4,1)=-0.8649;
	data(5,1)=-0.0301;
	data(6,1)=-0.1649;

	data(0,2)=0.6277;
	data(1,2)=1.0933;
	data(2,2)=1.1093;
	data(3,2)=-0.8637;
	data(4,2)=0.0774;
	data(5,2)=-1.2141;
	data(6,2)=-1.1135;

	SGVector<int32_t> lab(7);
	lab[0]=2;
	lab[1]=1;
	lab[2]=1;
	lab[3]=0;
	lab[4]=2;
	lab[5]=0;
	lab[6]=0;

	CMulticlassLabels* labels=new CMulticlassLabels();
	labels->set_int_labels(lab);

	CSoftMaxLikelihood* sml=new CSoftMaxLikelihood();
	SGVector<float64_t> data_vector=SGVector<float64_t>(data.matrix,data.num_rows*data.num_cols,false);
	SGVector<float64_t> v=sml->get_log_probability_derivative_f(labels,data_vector,1);

	float64_t ep=0.0001;
	EXPECT_NEAR(v[0],-0.1382,ep);
	EXPECT_NEAR(v[1],-0.0138,ep);
	EXPECT_NEAR(v[2],-0.4887,ep);
	EXPECT_NEAR(v[3],0.5638,ep);
	EXPECT_NEAR(v[4],-0.2384,ep);
	EXPECT_NEAR(v[5],0.2435,ep);
	EXPECT_NEAR(v[6],0.8669,ep);
	EXPECT_NEAR(v[7],-0.2803,ep);
	EXPECT_NEAR(v[8],0.7945,ep);
	EXPECT_NEAR(v[9],0.8404,ep);
	EXPECT_NEAR(v[10],-0.4309,ep);
	EXPECT_NEAR(v[11],-0.2136,ep);
	EXPECT_NEAR(v[12],-0.1865,ep);
	EXPECT_NEAR(v[13],-0.6249,ep);
	EXPECT_NEAR(v[14],0.4184,ep);
	EXPECT_NEAR(v[15],-0.7807,ep);
	EXPECT_NEAR(v[16],-0.3517,ep);
	EXPECT_NEAR(v[17],-0.1329,ep);
	EXPECT_NEAR(v[18],0.4520,ep);
	EXPECT_NEAR(v[19],-0.0571,ep);
	EXPECT_NEAR(v[20],-0.2420,ep);

	SG_UNREF(sml);
	SG_UNREF(labels);
}

TEST(SoftMaxLikelihood,get_log_derivatives_second)
{
	SGMatrix<float64_t> data(7,3);

	data(0,0)=-0.8095;
	data(1,0)=-2.9443;
	data(2,0)=1.4384;
	data(3,0)=0.3252;
	data(4,0)=-0.7549;
	data(5,0)=1.3703;
	data(6,0)=-1.7115;

	data(0,1)=-0.1022;
	data(1,1)=-0.2414;
	data(2,1)=0.3192;
	data(3,1)=0.3129;
	data(4,1)=-0.8649;
	data(5,1)=-0.0301;
	data(6,1)=-0.1649;

	data(0,2)=0.6277;
	data(1,2)=1.0933;
	data(2,2)=1.1093;
	data(3,2)=-0.8637;
	data(4,2)=0.0774;
	data(5,2)=-1.2141;
	data(6,2)=-1.1135;

	SGVector<int32_t> lab(7);
	lab[0]=2;
	lab[1]=1;
	lab[2]=1;
	lab[3]=0;
	lab[4]=2;
	lab[5]=0;
	lab[6]=0;

	CMulticlassLabels* labels=new CMulticlassLabels();
	labels->set_int_labels(lab);

	CSoftMaxLikelihood* sml=new CSoftMaxLikelihood();
	SGVector<float64_t> data_vector=SGVector<float64_t>(data.matrix,data.num_rows*data.num_cols,false);
	SGVector<float64_t> v=sml->get_log_probability_derivative_f(labels,data_vector,2);

	float64_t ep=0.0001;
	EXPECT_NEAR(v[0],-0.1191,ep);
	EXPECT_NEAR(v[1],0.0387,ep);
	EXPECT_NEAR(v[2],0.0804,ep);
	EXPECT_NEAR(v[3],-0.0136,ep);
	EXPECT_NEAR(v[4],0.0028,ep);
	EXPECT_NEAR(v[5],0.0108,ep);
	EXPECT_NEAR(v[6],-0.2499,ep);
	EXPECT_NEAR(v[7],0.0780,ep);
	EXPECT_NEAR(v[8],0.1719,ep);
	EXPECT_NEAR(v[9],-0.2459,ep);
	EXPECT_NEAR(v[10],0.1880,ep);
	EXPECT_NEAR(v[11],0.0580,ep);
	EXPECT_NEAR(v[12],-0.1816,ep);
	EXPECT_NEAR(v[13],0.0509,ep);
	EXPECT_NEAR(v[14],0.1306,ep);
	EXPECT_NEAR(v[15],-0.1842,ep);
	EXPECT_NEAR(v[16],0.1411,ep);
	EXPECT_NEAR(v[17],0.0432,ep);
	EXPECT_NEAR(v[18],-0.1154,ep);
	EXPECT_NEAR(v[19],0.0832,ep);
	EXPECT_NEAR(v[20],0.0322,ep);
	EXPECT_NEAR(v[21],0.0387,ep);
	EXPECT_NEAR(v[22],-0.2017,ep);
	EXPECT_NEAR(v[23],0.1630,ep);
	EXPECT_NEAR(v[24],0.0028,ep);
	EXPECT_NEAR(v[25],-0.1633,ep);
	EXPECT_NEAR(v[26],0.1604,ep);
	EXPECT_NEAR(v[27],0.0780,ep);
	EXPECT_NEAR(v[28],-0.1341,ep);
	EXPECT_NEAR(v[29],0.0561,ep);
	EXPECT_NEAR(v[30],0.1880,ep);
	EXPECT_NEAR(v[31],-0.2452,ep);
	EXPECT_NEAR(v[32],0.0573,ep);
	EXPECT_NEAR(v[33],0.0509,ep);
	EXPECT_NEAR(v[34],-0.1680,ep);
	EXPECT_NEAR(v[35],0.1170,ep);
	EXPECT_NEAR(v[36],0.1411,ep);
	EXPECT_NEAR(v[37],-0.1517,ep);
	EXPECT_NEAR(v[38],0.0106,ep);
	EXPECT_NEAR(v[39],0.0832,ep);
	EXPECT_NEAR(v[40],-0.2344,ep);
	EXPECT_NEAR(v[41],0.1512,ep);
	EXPECT_NEAR(v[42],0.0804,ep);
	EXPECT_NEAR(v[43],0.1630,ep);
	EXPECT_NEAR(v[44],-0.2433,ep);
	EXPECT_NEAR(v[45],0.0108,ep);
	EXPECT_NEAR(v[46],0.1604,ep);
	EXPECT_NEAR(v[47],-0.1712,ep);
	EXPECT_NEAR(v[48],0.1719,ep);
	EXPECT_NEAR(v[49],0.0561,ep);
	EXPECT_NEAR(v[50],-0.2280,ep);
	EXPECT_NEAR(v[51],0.0580,ep);
	EXPECT_NEAR(v[52],0.0573,ep);
	EXPECT_NEAR(v[53],-0.1152,ep);
	EXPECT_NEAR(v[54],0.1306,ep);
	EXPECT_NEAR(v[55],0.1170,ep);
	EXPECT_NEAR(v[56],-0.2477,ep);
	EXPECT_NEAR(v[57],0.0432,ep);
	EXPECT_NEAR(v[58],0.0106,ep);
	EXPECT_NEAR(v[59],-0.0538,ep);
	EXPECT_NEAR(v[60],0.0322,ep);
	EXPECT_NEAR(v[61],0.1512,ep);
	EXPECT_NEAR(v[62],-0.1834,ep);

	SG_UNREF(sml);
	SG_UNREF(labels);
}

TEST(SoftMaxLikelihood,get_log_derivatives_third)
{
	SGMatrix<float64_t> data(7,3);

	data(0,0)=-0.8095;
	data(1,0)=-2.9443;
	data(2,0)=1.4384;
	data(3,0)=0.3252;
	data(4,0)=-0.7549;
	data(5,0)=1.3703;
	data(6,0)=-1.7115;

	data(0,1)=-0.1022;
	data(1,1)=-0.2414;
	data(2,1)=0.3192;
	data(3,1)=0.3129;
	data(4,1)=-0.8649;
	data(5,1)=-0.0301;
	data(6,1)=-0.1649;

	data(0,2)=0.6277;
	data(1,2)=1.0933;
	data(2,2)=1.1093;
	data(3,2)=-0.8637;
	data(4,2)=0.0774;
	data(5,2)=-1.2141;
	data(6,2)=-1.1135;

	SGVector<int32_t> lab(7);
	lab[0]=2;
	lab[1]=1;
	lab[2]=1;
	lab[3]=0;
	lab[4]=2;
	lab[5]=0;
	lab[6]=0;

	CMulticlassLabels* labels=new CMulticlassLabels();
	labels->set_int_labels(lab);

	CSoftMaxLikelihood* sml=new CSoftMaxLikelihood();
	SGVector<float64_t> data_vector=SGVector<float64_t>(data.matrix,data.num_rows*data.num_cols,false);
	SGVector<float64_t> v=sml->get_log_probability_derivative_f(labels,data_vector,3);

	float64_t ep=0.0001;
	EXPECT_NEAR(v[0],0.0862,ep);
	EXPECT_NEAR(v[1],-0.0280,ep);
	EXPECT_NEAR(v[2],-0.0581,ep);
	EXPECT_NEAR(v[3],-0.0280,ep);
	EXPECT_NEAR(v[4],-0.0170,ep);
	EXPECT_NEAR(v[5],0.0450,ep);
	EXPECT_NEAR(v[6],-0.0581,ep);
	EXPECT_NEAR(v[7],0.0450,ep);
	EXPECT_NEAR(v[8],0.0131,ep);
	EXPECT_NEAR(v[9],-0.0280,ep);
	EXPECT_NEAR(v[10],-0.0170,ep);
	EXPECT_NEAR(v[11],0.0450,ep);
	EXPECT_NEAR(v[12],-0.0170,ep);
	EXPECT_NEAR(v[13],0.0886,ep);
	EXPECT_NEAR(v[14],-0.0716,ep);
	EXPECT_NEAR(v[15],0.0450,ep);
	EXPECT_NEAR(v[16],-0.0716,ep);
	EXPECT_NEAR(v[17],0.0266,ep);
	EXPECT_NEAR(v[18],-0.0581,ep);
	EXPECT_NEAR(v[19],0.0450,ep);
	EXPECT_NEAR(v[20],0.0131,ep);
	EXPECT_NEAR(v[21],0.0450,ep);
	EXPECT_NEAR(v[22],-0.0716,ep);
	EXPECT_NEAR(v[23],0.0266,ep);
	EXPECT_NEAR(v[24],0.0131,ep);
	EXPECT_NEAR(v[25],0.0266,ep);
	EXPECT_NEAR(v[26],-0.0397,ep);
	EXPECT_NEAR(v[27],0.0132,ep);
	EXPECT_NEAR(v[28],-0.0028,ep);
	EXPECT_NEAR(v[29],-0.0105,ep);
	EXPECT_NEAR(v[30],-0.0028,ep);
	EXPECT_NEAR(v[31],-0.0017,ep);
	EXPECT_NEAR(v[32],0.0044,ep);
	EXPECT_NEAR(v[33],-0.0105,ep);
	EXPECT_NEAR(v[34],0.0044,ep);
	EXPECT_NEAR(v[35],0.0060,ep);
	EXPECT_NEAR(v[36],-0.0028,ep);
	EXPECT_NEAR(v[37],-0.0017,ep);
	EXPECT_NEAR(v[38],0.0044,ep);
	EXPECT_NEAR(v[39],-0.0017,ep);
	EXPECT_NEAR(v[40],0.0962,ep);
	EXPECT_NEAR(v[41],-0.0945,ep);
	EXPECT_NEAR(v[42],0.0044,ep);
	EXPECT_NEAR(v[43],-0.0945,ep);
	EXPECT_NEAR(v[44],0.0901,ep);
	EXPECT_NEAR(v[45],-0.0105,ep);
	EXPECT_NEAR(v[46],0.0044,ep);
	EXPECT_NEAR(v[47],0.0060,ep);
	EXPECT_NEAR(v[48],0.0044,ep);
	EXPECT_NEAR(v[49],-0.0945,ep);
	EXPECT_NEAR(v[50],0.0901,ep);
	EXPECT_NEAR(v[51],0.0060,ep);
	EXPECT_NEAR(v[52],0.0901,ep);
	EXPECT_NEAR(v[53],-0.0961,ep);
	EXPECT_NEAR(v[54],0.0056,ep);
	EXPECT_NEAR(v[55],-0.0018,ep);
	EXPECT_NEAR(v[56],-0.0039,ep);
	EXPECT_NEAR(v[57],-0.0018,ep);
	EXPECT_NEAR(v[58],-0.0531,ep);
	EXPECT_NEAR(v[59],0.0549,ep);
	EXPECT_NEAR(v[60],-0.0039,ep);
	EXPECT_NEAR(v[61],0.0549,ep);
	EXPECT_NEAR(v[62],-0.0510,ep);
	EXPECT_NEAR(v[63],-0.0018,ep);
	EXPECT_NEAR(v[64],-0.0531,ep);
	EXPECT_NEAR(v[65],0.0549,ep);
	EXPECT_NEAR(v[66],-0.0531,ep);
	EXPECT_NEAR(v[67],0.0913,ep);
	EXPECT_NEAR(v[68],-0.0382,ep);
	EXPECT_NEAR(v[69],0.0549,ep);
	EXPECT_NEAR(v[70],-0.0382,ep);
	EXPECT_NEAR(v[71],-0.0166,ep);
	EXPECT_NEAR(v[72],-0.0039,ep);
	EXPECT_NEAR(v[73],0.0549,ep);
	EXPECT_NEAR(v[74],-0.0510,ep);
	EXPECT_NEAR(v[75],0.0549,ep);
	EXPECT_NEAR(v[76],-0.0382,ep);
	EXPECT_NEAR(v[77],-0.0166,ep);
	EXPECT_NEAR(v[78],-0.0510,ep);
	EXPECT_NEAR(v[79],-0.0166,ep);
	EXPECT_NEAR(v[80],0.0676,ep);
	EXPECT_NEAR(v[81],0.0314,ep);
	EXPECT_NEAR(v[82],-0.0240,ep);
	EXPECT_NEAR(v[83],-0.0074,ep);
	EXPECT_NEAR(v[84],-0.0240,ep);
	EXPECT_NEAR(v[85],-0.0260,ep);
	EXPECT_NEAR(v[86],0.0500,ep);
	EXPECT_NEAR(v[87],-0.0074,ep);
	EXPECT_NEAR(v[88],0.0500,ep);
	EXPECT_NEAR(v[89],-0.0426,ep);
	EXPECT_NEAR(v[90],-0.0240,ep);
	EXPECT_NEAR(v[91],-0.0260,ep);
	EXPECT_NEAR(v[92],0.0500,ep);
	EXPECT_NEAR(v[93],-0.0260,ep);
	EXPECT_NEAR(v[94],0.0339,ep);
	EXPECT_NEAR(v[95],-0.0079,ep);
	EXPECT_NEAR(v[96],0.0500,ep);
	EXPECT_NEAR(v[97],-0.0079,ep);
	EXPECT_NEAR(v[98],-0.0420,ep);
	EXPECT_NEAR(v[99],-0.0074,ep);
	EXPECT_NEAR(v[100],0.0500,ep);
	EXPECT_NEAR(v[101],-0.0426,ep);
	EXPECT_NEAR(v[102],0.0500,ep);
	EXPECT_NEAR(v[103],-0.0079,ep);
	EXPECT_NEAR(v[104],-0.0420,ep);
	EXPECT_NEAR(v[105],-0.0426,ep);
	EXPECT_NEAR(v[106],-0.0420,ep);
	EXPECT_NEAR(v[107],0.0846,ep);
	EXPECT_NEAR(v[108],0.0950,ep);
	EXPECT_NEAR(v[109],-0.0266,ep);
	EXPECT_NEAR(v[110],-0.0684,ep);
	EXPECT_NEAR(v[111],-0.0266,ep);
	EXPECT_NEAR(v[112],-0.0292,ep);
	EXPECT_NEAR(v[113],0.0558,ep);
	EXPECT_NEAR(v[114],-0.0684,ep);
	EXPECT_NEAR(v[115],0.0558,ep);
	EXPECT_NEAR(v[116],0.0125,ep);
	EXPECT_NEAR(v[117],-0.0266,ep);
	EXPECT_NEAR(v[118],-0.0292,ep);
	EXPECT_NEAR(v[119],0.0558,ep);
	EXPECT_NEAR(v[120],-0.0292,ep);
	EXPECT_NEAR(v[121],0.0962,ep);
	EXPECT_NEAR(v[122],-0.0670,ep);
	EXPECT_NEAR(v[123],0.0558,ep);
	EXPECT_NEAR(v[124],-0.0670,ep);
	EXPECT_NEAR(v[125],0.0112,ep);
	EXPECT_NEAR(v[126],-0.0684,ep);
	EXPECT_NEAR(v[127],0.0558,ep);
	EXPECT_NEAR(v[128],0.0125,ep);
	EXPECT_NEAR(v[129],0.0558,ep);
	EXPECT_NEAR(v[130],-0.0670,ep);
	EXPECT_NEAR(v[131],0.0112,ep);
	EXPECT_NEAR(v[132],0.0125,ep);
	EXPECT_NEAR(v[133],0.0112,ep);
	EXPECT_NEAR(v[134],-0.0238,ep);
	EXPECT_NEAR(v[135],-0.0945,ep);
	EXPECT_NEAR(v[136],0.0724,ep);
	EXPECT_NEAR(v[137],0.0221,ep);
	EXPECT_NEAR(v[138],0.0724,ep);
	EXPECT_NEAR(v[139],-0.0885,ep);
	EXPECT_NEAR(v[140],0.0161,ep);
	EXPECT_NEAR(v[141],0.0221,ep);
	EXPECT_NEAR(v[142],0.0161,ep);
	EXPECT_NEAR(v[143],-0.0382,ep);
	EXPECT_NEAR(v[144],0.0724,ep);
	EXPECT_NEAR(v[145],-0.0885,ep);
	EXPECT_NEAR(v[146],0.0161,ep);
	EXPECT_NEAR(v[147],-0.0885,ep);
	EXPECT_NEAR(v[148],0.0951,ep);
	EXPECT_NEAR(v[149],-0.0067,ep);
	EXPECT_NEAR(v[150],0.0161,ep);
	EXPECT_NEAR(v[151],-0.0067,ep);
	EXPECT_NEAR(v[152],-0.0094,ep);
	EXPECT_NEAR(v[153],0.0221,ep);
	EXPECT_NEAR(v[154],0.0161,ep);
	EXPECT_NEAR(v[155],-0.0382,ep);
	EXPECT_NEAR(v[156],0.0161,ep);
	EXPECT_NEAR(v[157],-0.0067,ep);
	EXPECT_NEAR(v[158],-0.0094,ep);
	EXPECT_NEAR(v[159],-0.0382,ep);
	EXPECT_NEAR(v[160],-0.0094,ep);
	EXPECT_NEAR(v[161],0.0477,ep);
	EXPECT_NEAR(v[162],0.0847,ep);
	EXPECT_NEAR(v[163],-0.0610,ep);
	EXPECT_NEAR(v[164],-0.0236,ep);
	EXPECT_NEAR(v[165],-0.0610,ep);
	EXPECT_NEAR(v[166],0.0208,ep);
	EXPECT_NEAR(v[167],0.0403,ep);
	EXPECT_NEAR(v[168],-0.0236,ep);
	EXPECT_NEAR(v[169],0.0403,ep);
	EXPECT_NEAR(v[170],-0.0166,ep);
	EXPECT_NEAR(v[171],-0.0610,ep);
	EXPECT_NEAR(v[172],0.0208,ep);
	EXPECT_NEAR(v[173],0.0403,ep);
	EXPECT_NEAR(v[174],0.0208,ep);
	EXPECT_NEAR(v[175],-0.0586,ep);
	EXPECT_NEAR(v[176],0.0378,ep);
	EXPECT_NEAR(v[177],0.0403,ep);
	EXPECT_NEAR(v[178],0.0378,ep);
	EXPECT_NEAR(v[179],-0.0780,ep);
	EXPECT_NEAR(v[180],-0.0236,ep);
	EXPECT_NEAR(v[181],0.0403,ep);
	EXPECT_NEAR(v[182],-0.0166,ep);
	EXPECT_NEAR(v[183],0.0403,ep);
	EXPECT_NEAR(v[184],0.0378,ep);
	EXPECT_NEAR(v[185],-0.0780,ep);
	EXPECT_NEAR(v[186],-0.0166,ep);
	EXPECT_NEAR(v[187],-0.0780,ep);
	EXPECT_NEAR(v[188],0.0947,ep);

	SG_UNREF(sml);
	SG_UNREF(labels);
}

#endif /* HAVE_EIGEN3 */
