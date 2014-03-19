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
#include <shogun/multiclass/tree/TreeMachineNode.h>
#include <shogun/multiclass/tree/ID3TreeNodeData.h>
#include <shogun/lib/DynamicObjectArray.h>

using namespace std;
using namespace shogun;

TEST(TreeMachineNode, build_tree)
{
	CTreeMachineNode<id3TreeNodeData>* root= 
			new CTreeMachineNode<id3TreeNodeData>();

	CTreeMachineNode<id3TreeNodeData>* child1= 
			new CTreeMachineNode<id3TreeNodeData>();
	child1->machine(1);
	child1->data.attribute_id=1;
	child1->data.transit_if_feature_value=1.0;
	child1->data.class_label=11.0;

	CTreeMachineNode<id3TreeNodeData>* child2= 
			new CTreeMachineNode<id3TreeNodeData>();

	root->add_child(child1);

	CDynamicObjectArray* insert_children=new CDynamicObjectArray();
	insert_children->push_back(child1);
	insert_children->push_back(child2);
	root->set_children(insert_children);

	CDynamicObjectArray* get_children=root->get_children();
	CTreeMachineNode<id3TreeNodeData>* get_child1=((CTreeMachineNode<id3TreeNodeData>*)
							 get_children->get_element(0));

	EXPECT_EQ(get_child1->data.attribute_id,1);
	EXPECT_EQ(get_child1->data.transit_if_feature_value,1.0);
	EXPECT_EQ(get_child1->data.class_label,11.0);
	EXPECT_EQ(get_child1->machine(),1);
	EXPECT_EQ(get_child1->parent()->machine(),-1);

	SG_UNREF(root);
	SG_UNREF(insert_children);
	SG_UNREF(get_children);
	SG_UNREF(get_child1);
}
