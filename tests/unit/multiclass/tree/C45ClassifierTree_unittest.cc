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
#include <shogun/multiclass/tree/C45ClassifierTree.h>
#include <gtest/gtest.h>

using namespace shogun;

#define sunny 1.
#define overcast 2.
#define rain 3.

#define hot 1.
#define mild 2.
#define cool 3.

#define high 1.
#define normal 2.

#define weak 1.
#define strong 2.

TEST(C45ClassifierTree, classify_equivalence_check_to_id3)
{
/* Example from  http://www.cise.ufl.edu/~ddd/cap6635/Fall-97/Short-papers/2.htm */

	SGMatrix<float64_t> data(4,15);

	//vector = [Outlook Temperature Humidity Wind]
	data(0,0)=sunny;
	data(1,0)=hot;
	data(2,0)=high;
	data(3,0)=weak;

	data(0,1)=sunny;
	data(1,1)=hot;
	data(2,1)=high;
	data(3,1)=strong;

	data(0,2)=overcast;
	data(1,2)=hot;
	data(2,2)=high;
	data(3,2)=weak;

	data(0,3)=rain;
	data(1,3)=mild;
	data(2,3)=high;
	data(3,3)=weak;

	data(0,4)=rain;
	data(1,4)=cool;
	data(2,4)=normal;
	data(3,4)=weak;

	data(0,5)=rain;
	data(1,5)=cool;
	data(2,5)=normal;
	data(3,5)=strong;

	data(0,6)=overcast;
	data(1,6)=cool;
	data(2,6)=normal;
	data(3,6)=strong;

	data(0,7)=sunny;
	data(1,7)=mild;
	data(2,7)=high;
	data(3,7)=weak;

	data(0,8)=sunny;
	data(1,8)=cool;
	data(2,8)=normal;
	data(3,8)=weak;

	data(0,9)=rain;
	data(1,9)=mild;
	data(2,9)=normal;
	data(3,9)=weak;

	data(0,10)=sunny;
	data(1,10)=mild;
	data(2,10)=normal;
	data(3,10)=strong;

	data(0,11)=overcast;
	data(1,11)=mild;
	data(2,11)=high;
	data(3,11)=strong;

	data(0,12)=overcast;
	data(1,12)=hot;
	data(2,12)=normal;
	data(3,12)=weak;

	data(0,13)=rain;
	data(1,13)=mild;
	data(2,13)=high;
	data(3,13)=strong;

	data(0,14)=overcast;
	data(1,14)=mild;
	data(2,14)=high;
	data(3,14)=strong;

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	// yes 1. no 0.
	SGVector<float64_t> lab(15);
	lab[0]=0.0;
	lab[1]=0.0;
	lab[2]=1.0;
	lab[3]=1.0;
	lab[4]=1.0;
	lab[5]=0.0;
	lab[6]=1.0;
	lab[7]=0.0;
	lab[8]=1.0;
	lab[9]=1.0;
	lab[10]=1.0;
	lab[11]=1.0;
	lab[12]=1.0;
	lab[13]=0.0;
	lab[14]=0.0;

	SGVector<bool> ft=SGVector<bool>(4);
	ft[0]=true;
	ft[1]=true;
	ft[2]=true;
	ft[3]=true;

	CMulticlassLabels* labels=new CMulticlassLabels(lab);

	CC45ClassifierTree* c45=new CC45ClassifierTree();
	c45->set_labels(labels);
	c45->set_feature_types(ft);
	c45->train(feats);

	SGMatrix<float64_t> test(4,5);
	test(0,0)=overcast;
	test(0,1)=rain;
	test(0,2)=sunny;
	test(0,3)=rain;
	test(0,4)=sunny;

	test(1,0)=hot;
	test(1,1)=cool;
	test(1,2)=mild;
	test(1,3)=mild;
	test(1,4)=hot;

	test(2,0)=normal;
	test(2,1)=high;
	test(2,2)=high;
	test(2,3)=normal;
	test(2,4)=normal;

	test(3,0)=strong;
	test(3,1)=strong;
	test(3,2)=weak;
	test(3,3)=weak;
	test(3,4)=strong;

	CDenseFeatures<float64_t>* test_feats=new CDenseFeatures<float64_t>(test);
	CMulticlassLabels* result=(CMulticlassLabels*) c45->apply(test_feats);
	SGVector<float64_t> res_vector=result->get_labels();

	EXPECT_EQ(1.0,res_vector[0]);
	EXPECT_EQ(0.0,res_vector[1]);
	EXPECT_EQ(0.0,res_vector[2]);
	EXPECT_EQ(1.0,res_vector[3]);
	EXPECT_EQ(1.0,res_vector[4]);	

	SG_UNREF(test_feats);
	SG_UNREF(result);
	SG_UNREF(c45);
	SG_UNREF(feats);
}

TEST(C45ClassifierTree, classify_continuous_plus_categorical_data)
{
/* Example from  http://www2.cs.uregina.ca/~dbd/cs831/notes/ml/dtrees/c4.5/c4.5_prob1.html */

	SGMatrix<float64_t> data(4,14);

	//vector = [Outlook Temperature Humidity Wind]
	data(0,0)=sunny;
	data(1,0)=85;
	data(2,0)=85;
	data(3,0)=0;

	data(0,1)=sunny;
	data(1,1)=80;
	data(2,1)=90;
	data(3,1)=1;

	data(0,2)=overcast;
	data(1,2)=83;
	data(2,2)=78;
	data(3,2)=0;

	data(0,3)=rain;
	data(1,3)=70;
	data(2,3)=96;
	data(3,3)=0;

	data(0,4)=rain;
	data(1,4)=68;
	data(2,4)=80;
	data(3,4)=0;

	data(0,5)=rain;
	data(1,5)=65;
	data(2,5)=70;
	data(3,5)=1;

	data(0,6)=overcast;
	data(1,6)=64;
	data(2,6)=65;
	data(3,6)=1;

	data(0,7)=sunny;
	data(1,7)=72;
	data(2,7)=95;
	data(3,7)=0;

	data(0,8)=sunny;
	data(1,8)=69;
	data(2,8)=70;
	data(3,8)=0;

	data(0,9)=rain;
	data(1,9)=75;
	data(2,9)=80;
	data(3,9)=0;

	data(0,10)=sunny;
	data(1,10)=75;
	data(2,10)=70;
	data(3,10)=1;

	data(0,11)=overcast;
	data(1,11)=72;
	data(2,11)=90;
	data(3,11)=1;

	data(0,12)=overcast;
	data(1,12)=81;
	data(2,12)=75;
	data(3,12)=0;

	data(0,13)=rain;
	data(1,13)=71;
	data(2,13)=80;
	data(3,13)=1;

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	// play 1. don't play 0.
	SGVector<float64_t> lab(14);
	lab[0]=0.0;
	lab[1]=0.0;
	lab[2]=1.0;
	lab[3]=1.0;
	lab[4]=1.0;
	lab[5]=0.0;
	lab[6]=1.0;
	lab[7]=0.0;
	lab[8]=1.0;
	lab[9]=1.0;
	lab[10]=1.0;
	lab[11]=1.0;
	lab[12]=1.0;
	lab[13]=0.0;

	SGVector<bool> ft=SGVector<bool>(4);
	ft[0]=true;
	ft[1]=false;
	ft[2]=false;
	ft[3]=true;

	CMulticlassLabels* labels=new CMulticlassLabels(lab);

	CC45ClassifierTree* c45=new CC45ClassifierTree();
	c45->set_labels(labels);
	c45->set_feature_types(ft);
	c45->train(feats);

	SGMatrix<float64_t> test(4,5);
	test(0,0)=overcast;
	test(0,1)=rain;
	test(0,2)=sunny;
	test(0,3)=rain;
	test(0,4)=sunny;

	test(1,0)=40;
	test(1,1)=70;
	test(1,2)=86;
	test(1,3)=92;
	test(1,4)=56;

	test(2,0)=50;
	test(2,1)=56;
	test(2,2)=79;
	test(2,3)=98;
	test(2,4)=63;

	test(3,0)=1;
	test(3,1)=1;
	test(3,2)=0;
	test(3,3)=0;
	test(3,4)=1;

	CDenseFeatures<float64_t>* test_feats=new CDenseFeatures<float64_t>(test);
	CMulticlassLabels* result=(CMulticlassLabels*) c45->apply(test_feats);
	SGVector<float64_t> res_vector=result->get_labels();

	EXPECT_EQ(1.0,res_vector[0]);
	EXPECT_EQ(0.0,res_vector[1]);
	EXPECT_EQ(0.0,res_vector[2]);
	EXPECT_EQ(1.0,res_vector[3]);
	EXPECT_EQ(1.0,res_vector[4]);	

	SG_UNREF(test_feats);
	SG_UNREF(result);
	SG_UNREF(c45);
	SG_UNREF(feats);
}

TEST(C45ClassifierTree, missing_attribute)
{
	SGMatrix<float64_t> data(1,8);

	data(0,0)=20.;
	data(0,1)=30.;
	data(0,2)=40.;
	data(0,3)=50.;
	data(0,4)=60.;
	data(0,5)=70.;
	data(0,6)=80.;
	data(0,7)=CC45ClassifierTree::MISSING;

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	SGVector<float64_t> lab(8);
	lab[0]=0.0;
	lab[1]=0.0;
	lab[2]=0.0;
	lab[3]=1.0;
	lab[4]=1.0;
	lab[5]=2.0;
	lab[6]=2.0;
	lab[7]=1.0;

	SGVector<bool> ft=SGVector<bool>(1);
	ft[0]=false;

	CMulticlassLabels* labels=new CMulticlassLabels(lab);

	CC45ClassifierTree* c45=new CC45ClassifierTree();
	c45->set_labels(labels);
	c45->set_feature_types(ft);
	c45->train(feats);

	SGMatrix<float64_t> test(1,2);
	test(0,0)=32;
	test(0,1)=75;

	CDenseFeatures<float64_t>* test_feats=new CDenseFeatures<float64_t>(test);
	CMulticlassLabels* result=(CMulticlassLabels*) c45->apply(test_feats);
	SGVector<float64_t> certainty=c45->get_certainty_vector();
	SGVector<float64_t> res_vector=result->get_labels();

	EXPECT_EQ(0.0,res_vector[0]);
	EXPECT_EQ(1.0,res_vector[1]);
	EXPECT_EQ(0.875,certainty[0]);
	EXPECT_EQ(0.5625,certainty[1]);

	SG_UNREF(test_feats);
	SG_UNREF(result);
	SG_UNREF(c45);
	SG_UNREF(feats);
}

TEST(C45ClassifierTree, tree_prune_categorical_attributes)
{

	// form toy data
	SGMatrix<float64_t> data(4,16);
	float64_t num=0.0;

	for (int32_t i=0;i<2;i++)
	{
		for (int32_t a=0;a<8;a++)
			data(0,i*8+a)=num;
	
		num++;
	}

	num=0.0;
	for (int32_t i=0;i<4;i++)
	{
		for (int32_t a=0;a<4;a++)
			data(1,i*4+a)=num;

		num=(num==1.0)?0.0:1.0;
	}

	num=0.0;
	for (int32_t i=0;i<8;i++)
	{
		for (int32_t a=0;a<2;a++)
			data(2,i*2+a)=num;

		num=(num==1.0)?0.0:1.0;
	}

	num=0.0;
	for (int32_t i=0;i<16;i++)
	{
		data(3,i)=num;
		num=(num==1.0)?0.0:1.0;
	}

	SGVector<bool> feature_types(4);
	feature_types[0]=true;
	feature_types[1]=true;
	feature_types[2]=true;
	feature_types[3]=true;

	// form toy labels
	SGVector<float64_t> train_labels(16);
	train_labels[0]=1;
	train_labels[1]=1;
	train_labels[2]=1;
	train_labels[3]=0;
	train_labels[4]=0;
	train_labels[5]=0;
	train_labels[6]=0;
	train_labels[7]=0;
	train_labels[8]=1;
	train_labels[9]=1;
	train_labels[10]=1;
	train_labels[11]=1;
	train_labels[12]=1;
	train_labels[13]=1;
	train_labels[14]=1;
	train_labels[15]=1;

	SGVector<float64_t> validation_labels(16);
	validation_labels[0]=1;
	validation_labels[1]=0;
	validation_labels[2]=0;
	validation_labels[3]=0;
	validation_labels[4]=1;
	validation_labels[5]=0;
	validation_labels[6]=0;
	validation_labels[7]=0;
	validation_labels[8]=1;
	validation_labels[9]=0;
	validation_labels[10]=1;
	validation_labels[11]=0;
	validation_labels[12]=1;
	validation_labels[13]=1;
	validation_labels[14]=1;
	validation_labels[15]=1;

	CDenseFeatures<float64_t>* train_features=new CDenseFeatures<float64_t>(data);
	CMulticlassLabels* train_lab=new CMulticlassLabels(train_labels);
	CMulticlassLabels* validation_lab=new CMulticlassLabels(validation_labels);

	CC45ClassifierTree* c45tree=new CC45ClassifierTree();
	c45tree->set_labels(train_lab);
	c45tree->set_feature_types(feature_types);
	c45tree->train(train_features);
	c45tree->prune_tree(train_features,validation_lab);

	CMulticlassLabels* result=(CMulticlassLabels*) c45tree->apply(train_features);
	SGVector<float64_t> res_vector=result->get_labels();

	EXPECT_EQ(1.0,res_vector[0]);
	EXPECT_EQ(0.0,res_vector[1]);
	EXPECT_EQ(1.0,res_vector[2]);
	EXPECT_EQ(0.0,res_vector[3]);
	EXPECT_EQ(0.0,res_vector[4]);
	EXPECT_EQ(0.0,res_vector[5]);
	EXPECT_EQ(0.0,res_vector[6]);
	EXPECT_EQ(0.0,res_vector[7]);
	EXPECT_EQ(1.0,res_vector[8]);
	EXPECT_EQ(1.0,res_vector[9]);
	EXPECT_EQ(1.0,res_vector[10]);
	EXPECT_EQ(1.0,res_vector[11]);
	EXPECT_EQ(1.0,res_vector[12]);
	EXPECT_EQ(1.0,res_vector[13]);
	EXPECT_EQ(1.0,res_vector[14]);
	EXPECT_EQ(1.0,res_vector[15]);

	SG_UNREF(train_features);
	SG_UNREF(validation_lab);
	SG_UNREF(result);
	SG_UNREF(c45tree);
}

TEST(C45ClassifierTree, tree_prune_continuous_attributes)
{

	// form toy data
	SGMatrix<float64_t> data(2,8);
	data(0,0)=20;
	data(0,1)=30;
	data(0,2)=40;
	data(0,3)=50;
	data(0,4)=60;
	data(0,5)=70;
	data(0,6)=80;
	data(0,7)=90;
	data(1,0)=10;
	data(1,1)=20;
	data(1,2)=30;
	data(1,3)=10;
	data(1,4)=20;
	data(1,5)=30;
	data(1,6)=40;
	data(1,7)=50;

	SGVector<bool> feature_types(2);
	feature_types[0]=false;
	feature_types[1]=false;

	// form toy labels
	SGVector<float64_t> train_labels(8);
	train_labels[0]=1;
	train_labels[1]=1;
	train_labels[2]=1;
	train_labels[3]=2;
	train_labels[4]=2;
	train_labels[5]=3;
	train_labels[6]=3;
	train_labels[7]=2;

	SGMatrix<float64_t> validation_data(2,3);
	validation_data(0,0)=75;
	validation_data(0,1)=78;
	validation_data(0,2)=33;
	validation_data(1,0)=33;
	validation_data(1,1)=44;
	validation_data(1,2)=21;

	SGVector<float64_t> validation_labels(3);
	validation_labels[0]=2;
	validation_labels[1]=2;
	validation_labels[2]=1;

	CDenseFeatures<float64_t>* train_features=new CDenseFeatures<float64_t>(data);
	CMulticlassLabels* train_lab=new CMulticlassLabels(train_labels);
	CDenseFeatures<float64_t>* validation_features=new CDenseFeatures<float64_t>(validation_data);
	CMulticlassLabels* validation_lab=new CMulticlassLabels(validation_labels);

	CC45ClassifierTree* c45tree=new CC45ClassifierTree();
	c45tree->set_labels(train_lab);
	c45tree->set_feature_types(feature_types);
	c45tree->train(train_features);
	c45tree->prune_tree(validation_features,validation_lab);

	CMulticlassLabels* result=(CMulticlassLabels*) c45tree->apply(train_features);
	SGVector<float64_t> res_vector=result->get_labels();

	EXPECT_EQ(1.0,res_vector[0]);
	EXPECT_EQ(1.0,res_vector[1]);
	EXPECT_EQ(1.0,res_vector[2]);
	EXPECT_EQ(2.0,res_vector[3]);
	EXPECT_EQ(2.0,res_vector[4]);
	EXPECT_EQ(2.0,res_vector[5]);
	EXPECT_EQ(2.0,res_vector[6]);
	EXPECT_EQ(2.0,res_vector[7]);

	SG_UNREF(train_features);
	SG_UNREF(validation_features);
	SG_UNREF(validation_lab);
	SG_UNREF(result);
	SG_UNREF(c45tree);
}
