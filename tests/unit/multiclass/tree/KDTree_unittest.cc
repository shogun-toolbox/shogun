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
#include <shogun/multiclass/tree/KDTree.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(KDTree,tree_structure)
{
	SGMatrix<float64_t> data(2,4);
	data(0,0)=-4;
	data(1,0)=0;
	data(0,1)=4;
	data(1,1)=0;
	data(0,2)=-1;
	data(1,2)=2;
	data(0,3)=1;
	data(1,3)=-2;

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	CKDTree* tree=new CKDTree();
	tree->build_tree(feats);

	CBinaryTreeMachineNode<KDTreeNodeData>* node=dynamic_cast<CBinaryTreeMachineNode<KDTreeNodeData>*>(tree->get_root());

	EXPECT_EQ(0,node->data.start_idx);
	EXPECT_EQ(3,node->data.end_idx);
	EXPECT_EQ(4,node->data.bbox_upper[0]);
	EXPECT_EQ(-4,node->data.bbox_lower[0]);
	EXPECT_EQ(2,node->data.bbox_upper[1]);
	EXPECT_EQ(-2,node->data.bbox_lower[1]);

	CBinaryTreeMachineNode<KDTreeNodeData>* child=node->left();

	EXPECT_EQ(0,child->data.start_idx);
	EXPECT_EQ(1,child->data.end_idx);
	EXPECT_EQ(-1,child->data.bbox_upper[0]);
	EXPECT_EQ(-4,child->data.bbox_lower[0]);
	EXPECT_EQ(2,child->data.bbox_upper[1]);
	EXPECT_EQ(0,child->data.bbox_lower[1]);

	SG_UNREF(child);
	child=node->right();

	EXPECT_EQ(2,child->data.start_idx);
	EXPECT_EQ(3,child->data.end_idx);
	EXPECT_EQ(4,child->data.bbox_upper[0]);
	EXPECT_EQ(1,child->data.bbox_lower[0]);
	EXPECT_EQ(0,child->data.bbox_upper[1]);
	EXPECT_EQ(-2,child->data.bbox_lower[1]);

	SG_UNREF(node);
	SG_UNREF(tree);
	SG_UNREF(feats);
	SG_UNREF(child);
}

TEST(KDTree, knn_query)
{
	SGMatrix<float64_t> data(2,4);
	data(0,0)=2;
	data(1,0)=0;
	data(0,1)=4;
	data(1,1)=0;
	data(0,2)=-3;
	data(1,2)=0;
	data(0,3)=0;
	data(1,3)=1;

	CDenseFeatures<float64_t>* feats=new CDenseFeatures<float64_t>(data);

	CKDTree* tree=new CKDTree();
	tree->build_tree(feats);

	SGMatrix<float64_t> test_data(2,1);
	test_data(0,0)=0;
	test_data(1,0)=0;

	CDenseFeatures<float64_t>* qfeats=new CDenseFeatures<float64_t>(test_data);
	tree->query_knn(qfeats,3);

	SGMatrix<index_t> ind=tree->get_knn_indices();

	EXPECT_EQ(3,ind(0,0));
	EXPECT_EQ(0,ind(1,0));
	EXPECT_EQ(2,ind(2,0));		

	SG_UNREF(qfeats);
	SG_UNREF(feats);
	SG_UNREF(tree);  		
}