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

#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/regression/LeastAngleRegression.h>
#include <shogun/labels/RegressionLabels.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LeastAngleRegression, lars_N_GreaterThan_D)
{
	SGMatrix<float64_t> data(3,5);
	float64_t array[15]={0.044550005575722, -0.433969606728583, -0.397935396933392, -0.778754072066602, -0.620105076569903, 				-0.542538248707627, 0.334313094513960, 0.421985645755003, 0.263031426076997, 0.516043376162584, 	
				0.159041471773470, 0.691318725364356, -0.116152404185664, 0.473047565770014, -0.013876505800334};
	memcpy(data.matrix,array,15*sizeof(float64_t));
	SGVector<float64_t> lab(5);
	lab[0]=-0.196155100498902;
	lab[1]=-5.376485285422094;
	lab[2]=-1.717489351713958;
	lab[3]=4.506538567065213;
	lab[4]=2.783591170569741;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	SG_REF(features);
	CRegressionLabels* labels=new CRegressionLabels(lab);
	SG_REF(labels);	
	CLeastAngleRegression* lars=new CLeastAngleRegression();
	lars->set_labels((CLabels*) labels);
	lars->train(features);

	SGVector<float64_t> active3= SGVector<float64_t>(lars->get_w_for_var(3));
	SGVector<float64_t> active2= SGVector<float64_t>(lars->get_w_for_var(2));
	SGVector<float64_t> active1= SGVector<float64_t>(lars->get_w_for_var(1));

	EXPECT_NEAR(active3[0],2.911072069591,0.000000000001);
	EXPECT_NEAR(active3[1],1.290672330338,0.000000000001);
	EXPECT_NEAR(active3[2],2.208741384416,0.000000000001);

	EXPECT_NEAR(active2[0],1.747958837898,0.000000000001);
	EXPECT_NEAR(active2[1],0.000000000000,0.000000000001);
	EXPECT_NEAR(active2[2],1.840553057519,0.000000000001);

	EXPECT_NEAR(active1[0],0.000000000000,0.000000000001);
	EXPECT_NEAR(active1[1],0.000000000000,0.000000000001);
	EXPECT_NEAR(active1[2],0.092594219621,0.000000000001);

	SG_UNREF(lars);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(LeastAngleRegression, lars_N_LessThan_D)
{
	SGMatrix<float64_t> data(5,3);
	float64_t array[15]={0.217778502400306, 0.660755393455389, 0.492143169178889, 0.668618163874328, 0.806098163441828, 
				-0.790379818206924, -0.745771163834136, -0.810293460958058, -0.740156729710306, -0.515540473266151, 
				0.572601315806618, 0.085015770378747, 0.318150291779169, 0.071538565835978, -0.290557690175677};
	memcpy(data.matrix,array,15*sizeof(float64_t));

	SGVector<float64_t> lab(3);
	lab[0]=3.771471612608209;
	lab[1]=-3.218048715328546;
	lab[2]=-0.553422897279663;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	SG_REF(features);
	CRegressionLabels* labels=new CRegressionLabels(lab);
	SG_REF(labels);	
	CLeastAngleRegression* lars=new CLeastAngleRegression();
	lars->set_labels((CLabels*) labels);
	lars->train(features);

	SGVector<float64_t> active2= SGVector<float64_t>(lars->get_w_for_var(2));
	SGVector<float64_t> active1= SGVector<float64_t>(lars->get_w_for_var(1));

	EXPECT_NEAR(active1[0],0.000000000000,0.000000000001);
	EXPECT_NEAR(active1[1],0.000000000000,0.000000000001);
	EXPECT_NEAR(active1[2],0.000000000000,0.000000000001);
	EXPECT_NEAR(active1[3],0.039226231353,0.000000000001);
	EXPECT_NEAR(active1[4],0.000000000000,0.000000000001);

	EXPECT_NEAR(active2[0],0.000000000000,0.000000000001);
	EXPECT_NEAR(active2[1],0.000000000000,0.000000000001);
	EXPECT_NEAR(active2[2],0.000000000000,0.000000000001);
	EXPECT_NEAR(active2[3],2.578863294056,0.000000000001);
	EXPECT_NEAR(active2[4],2.539637062702,0.000000000001);

	SG_UNREF(lars);
	SG_UNREF(features);
	SG_UNREF(labels);
}
