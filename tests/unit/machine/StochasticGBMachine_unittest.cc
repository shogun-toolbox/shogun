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
#include <shogun/labels/RegressionLabels.h>
#include <shogun/loss/SquaredLoss.h>
#include <shogun/machine/StochasticGBMachine.h>
#include <shogun/multiclass/tree/CARTree.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <gtest/gtest.h>

using namespace shogun;

SGMatrix<float64_t> get_sinusoid_samples(int32_t num_samples, SGVector<float64_t> labels)
{
	SGMatrix<float64_t> ret(1,num_samples);
	SGVector<float64_t>::random_vector(ret.matrix,num_samples,0,15);

	for (int32_t i=0;i<num_samples;i++)
		labels[i]=CMath::sin(ret(0,i));

	return ret;
}

TEST(StochasticGBMachine,sinusoid_curve_fitting)
{
	sg_rand->set_seed(10);

	int32_t num_train_samples=100;
	SGVector<float64_t> lab(num_train_samples);
	SGMatrix<float64_t> data=get_sinusoid_samples(num_train_samples,lab);
	CDenseFeatures<float64_t>* train_feats=new CDenseFeatures<float64_t>(data);
	CRegressionLabels* train_labels=new CRegressionLabels(lab);

	SGVector<float64_t> tlab(10);
	SGMatrix<float64_t> tdata(1,10);

	tlab[0]=-0.999585752311506259;
	tlab[1]=0.75965469336929492;
	tlab[2]=-0.425832103506531334;
	tlab[3]=0.298135616000050285;
	tlab[4]=-0.48828775732556795;
	tlab[5]=-0.031677813420380535;
	tlab[6]=0.144672857935527394;
	tlab[7]=-0.0810247683026898424;
	tlab[8]=-0.767723534099077121;
	tlab[9]=0.639868456911451666;

	tdata(0,0)=10.9667896982205075;
	tdata(0,1)=0.862781976084872615;
	tdata(0,2)=12.1264892751645501;
	tdata(0,3)=9.12203911322216499;
	tdata(0,4)=9.93490458930258313;
	tdata(0,5)=6.25150219333625934;
	tdata(0,6)=0.145182344164974608;
	tdata(0,7)=3.22270633960671393;
	tdata(0,8)=11.6910897047936668;
	tdata(0,9)=2.44726557225158103;

	CDenseFeatures<float64_t>* test_feats=new CDenseFeatures<float64_t>(tdata);
	CRegressionLabels* test_labels=new CRegressionLabels(tlab);

	SGVector<bool> ft(1);
	ft[0]=false;
	CCARTree* tree=new CCARTree(ft);
	tree->set_max_depth(2);
	CSquaredLoss* sq=new CSquaredLoss();

	CStochasticGBMachine* sgbm=new CStochasticGBMachine(tree,sq,100,0.1,1.0);
	sgbm->set_labels(train_labels);
	sgbm->train(train_feats);

	CRegressionLabels* ret_labels=sgbm->apply_regression(test_feats);
	SGVector<float64_t> ret=ret_labels->get_labels();

	float64_t epsilon=1e-8;
	EXPECT_NEAR(ret[0],-0.943157980,epsilon);
	EXPECT_NEAR(ret[1],0.769725470,epsilon);
	EXPECT_NEAR(ret[2],-0.065691733,epsilon);
	EXPECT_NEAR(ret[3],0.251266829,epsilon);
	EXPECT_NEAR(ret[4],-0.577155330,epsilon);
	EXPECT_NEAR(ret[5],0.113875818,epsilon);
	EXPECT_NEAR(ret[6],0.427405429,epsilon);
	EXPECT_NEAR(ret[7],-0.098310066,epsilon);
	EXPECT_NEAR(ret[8],-0.416565932,epsilon);
	EXPECT_NEAR(ret[9],0.542023083,epsilon);

	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(test_labels);
	SG_UNREF(ret_labels);
	SG_UNREF(sgbm);
}
