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

#include <gtest/gtest.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/multiclass/tree/BallTree.h>

using namespace shogun;

TEST(BallTree,tree_structure)
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

	auto feats=std::make_shared<DenseFeatures<float64_t>>(data);

	auto tree=std::make_shared<BallTree>();
	tree->build_tree(feats);

	auto node=tree->get_root()->as<BinaryTreeMachineNode<NbodyTreeNodeData>>();

	EXPECT_EQ(0,node->data.start_idx);
	EXPECT_EQ(3,node->data.end_idx);
	EXPECT_EQ(0,node->data.center[0]);
	EXPECT_EQ(0,node->data.center[1]);
	EXPECT_EQ(4,node->data.radius);

	auto child=node->left();

	EXPECT_EQ(0,child->data.start_idx);
	EXPECT_EQ(1,child->data.end_idx);
	EXPECT_EQ(-2.5,child->data.center[0]);
	EXPECT_EQ(1,child->data.center[1]);

	child=node->right();

	EXPECT_EQ(2,child->data.start_idx);
	EXPECT_EQ(3,child->data.end_idx);
	EXPECT_EQ(2.5,child->data.center[0]);
	EXPECT_EQ(-1,child->data.center[1]);

}

TEST(BallTree, knn_query)
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

	auto feats=std::make_shared<DenseFeatures<float64_t>>(data);

	auto tree=std::make_shared<BallTree>();
	tree->build_tree(feats);

	SGMatrix<float64_t> test_data(2,1);
	test_data(0,0)=0;
	test_data(1,0)=0;

	auto qfeats=std::make_shared<DenseFeatures<float64_t>>(test_data);
	tree->query_knn(qfeats,3);

	SGMatrix<index_t> ind=tree->get_knn_indices();

	EXPECT_EQ(3,ind(0,0));
	EXPECT_EQ(0,ind(1,0));
	EXPECT_EQ(2,ind(2,0));




}
