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
#include <shogun/multiclass/tree/BinaryTreeMachineNode.h>
#include <shogun/multiclass/tree/ID3TreeNodeData.h>

using namespace std;
using namespace shogun;

TEST(BinaryTreeMachineNode, build_tree)
{
	CBinaryTreeMachineNode<id3TreeNodeData>* root=
			new CBinaryTreeMachineNode<id3TreeNodeData>();

	CBinaryTreeMachineNode<id3TreeNodeData>* child1=
			new CBinaryTreeMachineNode<id3TreeNodeData>();

	CBinaryTreeMachineNode<id3TreeNodeData>* child2=
			new CBinaryTreeMachineNode<id3TreeNodeData>();
	child2->machine(2);
	child2->data.attribute_id=2;
	child2->data.transit_if_feature_value=2.0;
	child2->data.class_label=22.0;

	CBinaryTreeMachineNode<id3TreeNodeData>* child3=
			new CBinaryTreeMachineNode<id3TreeNodeData>();

	root->left(child1);
	root->right(child2);
	root->left(child2);
	root->right(child3);

	CBinaryTreeMachineNode<id3TreeNodeData>* get_left=root->left();
	CBinaryTreeMachineNode<id3TreeNodeData>* get_right=root->right();

	EXPECT_EQ(get_left->data.attribute_id,2);
	EXPECT_EQ(get_left->data.transit_if_feature_value,2.0);
	EXPECT_EQ(get_left->data.class_label,22.0);
	EXPECT_EQ(get_left->machine(),2);
	EXPECT_EQ(get_left->parent()->machine(),-1);

	SG_UNREF(root);
	SG_UNREF(get_left);
	SG_UNREF(get_right);
}
