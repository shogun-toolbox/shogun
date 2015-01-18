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
#include <shogun/multiclass/tree/CARTree.h>
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

TEST(CARTree, classify_nominal)
{
	SGMatrix<float64_t> data(4,14);

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

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	// yes 1. no 0.
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
	ft[1]=true;
	ft[2]=true;
	ft[3]=true;

	CMulticlassLabels* labels=new CMulticlassLabels(lab);

	CCARTree* c=new CCARTree();
	c->set_labels(labels);
	c->set_feature_types(ft);
	c->train(feats);

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
	CMulticlassLabels* result=(CMulticlassLabels*) c->apply(test_feats);
	SGVector<float64_t> res_vector=result->get_labels();

	EXPECT_EQ(1.0,res_vector[0]);
	EXPECT_EQ(0.0,res_vector[1]);
	EXPECT_EQ(0.0,res_vector[2]);
	EXPECT_EQ(1.0,res_vector[3]);
	EXPECT_EQ(1.0,res_vector[4]);	

	SG_UNREF(test_feats);
	SG_UNREF(result);
	SG_UNREF(c);
	SG_UNREF(feats);
}

TEST(CARTree, classify_non_nominal)
{
	SGMatrix<float64_t> data(4,14);

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

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	// yes 1. no 0.
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
	ft[0]=false;
	ft[1]=false;
	ft[2]=false;
	ft[3]=false;

	CMulticlassLabels* labels=new CMulticlassLabels(lab);

	CCARTree* c=new CCARTree();
	c->set_labels(labels);
	c->set_feature_types(ft);
	c->train(feats);

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
	CMulticlassLabels* result=(CMulticlassLabels*) c->apply(test_feats);
	SGVector<float64_t> res_vector=result->get_labels();

	EXPECT_EQ(1.0,res_vector[0]);
	EXPECT_EQ(0.0,res_vector[1]);
	EXPECT_EQ(0.0,res_vector[2]);
	EXPECT_EQ(1.0,res_vector[3]);
	EXPECT_EQ(1.0,res_vector[4]);	

	SG_UNREF(test_feats);
	SG_UNREF(result);
	SG_UNREF(c);
	SG_UNREF(feats);
}

TEST(CARTree, handle_missing_nominal)
{
	SGMatrix<float64_t> data(3,9);
	data(0,0)=1;
	data(1,0)=3;
	data(2,0)=6;

	data(0,1)=1;
	data(1,1)=3;
	data(2,1)=8;

	data(0,2)=1;
	data(1,2)=3;
	data(2,2)=6;

	data(0,3)=2;
	data(1,3)=5;
	data(2,3)=7;

	data(0,4)=2;
	data(1,4)=4;
	data(2,4)=8;

	data(0,5)=CCARTree::MISSING;
	data(1,5)=5;
	data(2,5)=8;

	data(0,6)=3;
	data(1,6)=4;
	data(2,6)=8;

	data(0,7)=3;
	data(1,7)=4;
	data(2,7)=8;

	data(0,8)=3;
	data(1,8)=4;
	data(2,8)=8;

	SGVector<float64_t> lab(9);
	lab[0]=1;
	lab[1]=1;
	lab[2]=1;
	lab[3]=1;
	lab[4]=1;
	lab[5]=1;
	lab[6]=2;
	lab[7]=2;
	lab[8]=2;

	SGVector<bool> ft=SGVector<bool>(3);
	ft[0]=true;
	ft[1]=true;
	ft[2]=true;

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);
	CMulticlassLabels* labels=new CMulticlassLabels(lab);

	CCARTree* c=new CCARTree();
	c->set_labels(labels);
	c->set_feature_types(ft);
	c->train(feats);

	CBinaryTreeMachineNode<CARTreeNodeData>* root=dynamic_cast<CBinaryTreeMachineNode<CARTreeNodeData>*>(c->get_root());
	CBinaryTreeMachineNode<CARTreeNodeData>* left=root->left();
	CBinaryTreeMachineNode<CARTreeNodeData>* right=root->right();

	EXPECT_EQ(0.0,root->data.attribute_id);
	EXPECT_EQ(9.0,root->data.total_weight);
	EXPECT_EQ(6.0,left->data.total_weight);
	EXPECT_EQ(3.0,right->data.total_weight);

	SG_UNREF(root);
	SG_UNREF(left);
	SG_UNREF(right);
	SG_UNREF(c);
	SG_UNREF(feats);
}

TEST(CARTree, handle_missing_continuous)
{
	SGMatrix<float64_t> data(3,9);
	data(0,0)=1;
	data(1,0)=3;
	data(2,0)=6;

	data(0,1)=1;
	data(1,1)=3;
	data(2,1)=6;

	data(0,2)=1;
	data(1,2)=3;
	data(2,2)=6;

	data(0,3)=2;
	data(1,3)=5;
	data(2,3)=7;

	data(0,4)=2;
	data(1,4)=4;
	data(2,4)=8;

	data(0,5)=CCARTree::MISSING;
	data(1,5)=5;
	data(2,5)=7;

	data(0,6)=3;
	data(1,6)=4;
	data(2,6)=8;

	data(0,7)=3;
	data(1,7)=4;
	data(2,7)=8;

	data(0,8)=3;
	data(1,8)=4;
	data(2,8)=8;

	SGVector<float64_t> lab(9);
	lab[0]=1;
	lab[1]=1;
	lab[2]=1;
	lab[3]=1;
	lab[4]=1;
	lab[5]=1;
	lab[6]=2;
	lab[7]=2;
	lab[8]=2;

	SGVector<bool> ft=SGVector<bool>(3);
	ft[0]=false;
	ft[1]=false;
	ft[2]=false;

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);
	CMulticlassLabels* labels=new CMulticlassLabels(lab);

	CCARTree* c=new CCARTree();
	c->set_labels(labels);
	c->set_feature_types(ft);
	c->train(feats);

	CBinaryTreeMachineNode<CARTreeNodeData>* root=dynamic_cast<CBinaryTreeMachineNode<CARTreeNodeData>*>(c->get_root());
	CBinaryTreeMachineNode<CARTreeNodeData>* left=root->left();
	CBinaryTreeMachineNode<CARTreeNodeData>* right=root->right();

	EXPECT_EQ(2.0,root->data.attribute_id);
	EXPECT_EQ(9.0,root->data.total_weight);
	EXPECT_EQ(5.0,left->data.total_weight);
	EXPECT_EQ(4.0,right->data.total_weight);

	SG_UNREF(root);
	SG_UNREF(left);
	SG_UNREF(right);
	SG_UNREF(c);
	SG_UNREF(feats);
}

TEST(CARTree, form_t1_test)
{
	SGMatrix<float64_t> data(1,5);
	data(0,0)=1;
	data(0,1)=1;
	data(0,2)=1;
	data(0,3)=2;
	data(0,4)=2;

	SGVector<float64_t> lab(5);
	lab[0]=0;
	lab[1]=1;
	lab[2]=1;
	lab[3]=1;
	lab[4]=0;

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	SGVector<bool> ft=SGVector<bool>(1);
	ft[0]=true;

	CMulticlassLabels* labels=new CMulticlassLabels(lab);
	CCARTree* c=new CCARTree();
	c->set_labels(labels);
	c->set_feature_types(ft);
	c->train(feats);

	CTreeMachineNode<CARTreeNodeData>* root=c->get_root();
	EXPECT_EQ(2,root->data.num_leaves);

	c->prune_using_test_dataset(feats,labels);

	SG_UNREF(root);
	root=c->get_root();
	EXPECT_EQ(1,root->data.num_leaves);

	SG_UNREF(c);
	SG_UNREF(feats);
	SG_UNREF(root);
}

TEST(CARTree,cv_prune_simple)
{
	sg_rand->set_seed(10);
	SGMatrix<float64_t> data(2,20);
	data(0,0)=2;
	data(1,0)=2;
	data(0,1)=1;
	data(1,1)=1;
	data(0,2)=2;
	data(1,2)=1;
	data(0,3)=2;
	data(1,3)=1;
	data(0,4)=2;
	data(1,4)=1;
	data(0,5)=2;
	data(1,5)=2;
	data(0,6)=2;
	data(1,6)=1;
	data(0,7)=2;
	data(1,7)=1;
	data(0,8)=2;
	data(1,8)=2;
	data(0,9)=2;
	data(1,9)=1;
	data(0,10)=2;
	data(1,10)=2;
	data(0,11)=2;
	data(1,11)=1;
	data(0,12)=2;
	data(1,12)=2;
	data(0,13)=2;
	data(1,13)=2;
	data(0,14)=2;
	data(1,14)=2;
	data(0,15)=2;
	data(1,15)=1;
	data(0,16)=1;
	data(1,16)=2;
	data(0,17)=1;
	data(1,17)=1;
	data(0,18)=1;
	data(1,18)=2;
	data(0,19)=1;
	data(1,19)=1;

	SGVector<float64_t> lab(20);
	lab[0]=0.0;
	lab[1]=1.0;
	lab[2]=1.0;
	lab[3]=1.0;
	lab[4]=1.0;
	lab[5]=0.0;
	lab[6]=1.0;
	lab[7]=1.0;
	lab[8]=0.0;
	lab[9]=1.0;
	lab[10]=0.0;
	lab[11]=0.0;
	lab[12]=0.0;
	lab[13]=0.0;
	lab[14]=0.0;
	lab[15]=0.0;
	lab[16]=1.0;
	lab[17]=1.0;
	lab[18]=1.0;
	lab[19]=1.0;

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	SGVector<bool> ft=SGVector<bool>(2);
	ft[0]=true;
	ft[1]=true;

	CMulticlassLabels* labels=new CMulticlassLabels(lab);
	CCARTree* c=new CCARTree();
	c->set_labels(labels);
	c->set_feature_types(ft);
	c->train(feats);

	CBinaryTreeMachineNode<CARTreeNodeData>* root=dynamic_cast<CBinaryTreeMachineNode<CARTreeNodeData>*>(c->get_root());

	EXPECT_EQ(4,root->data.num_leaves);
	EXPECT_EQ(2.0,root->data.weight_minus_branch);

	c->set_num_folds(2);
	c->set_cv_pruning();
	c->train(feats);

	SG_UNREF(root);
	root=dynamic_cast<CBinaryTreeMachineNode<CARTreeNodeData>*>(c->get_root());

	EXPECT_EQ(3,root->data.num_leaves);
	EXPECT_EQ(2.0,root->data.weight_minus_branch);

	SG_UNREF(c);
	SG_UNREF(feats);
	SG_UNREF(root);
}
