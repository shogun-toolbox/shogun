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
#include <shogun/multiclass/tree/TreeMachine.h>
#include <shogun/multiclass/tree/ID3TreeNodeData.h>
#include <shogun/lib/DynamicObjectArray.h>

using namespace std;
using namespace shogun;

TEST(TreeMachine, ID3TreeNodeData)
{
	CTreeMachineNode<id3TreeNodeData>* root= 
			new CTreeMachineNode<id3TreeNodeData>();
	SG_REF(root);
	
	CTreeMachine<id3TreeNodeData>* tree=
			new CTreeMachine<id3TreeNodeData>();
	tree->set_root(root);
	
	SG_UNREF(root);

	CTreeMachineNode<id3TreeNodeData>* child1= 
			new CTreeMachineNode<id3TreeNodeData>();

	CTreeMachineNode<id3TreeNodeData>* child2= 
			new CTreeMachineNode<id3TreeNodeData>();
	child2->machine(2);
	child2->data.attribute_id=2;
	child2->data.transit_if_feature_value=2.0;
	child2->data.class_label=22.0;

	CTreeMachineNode<id3TreeNodeData>* child3= 
			new CTreeMachineNode<id3TreeNodeData>();

	CDynamicObjectArray* insert_children = new CDynamicObjectArray();
	insert_children->push_back(child1);
	insert_children->push_back(child2);

	CTreeMachineNode<id3TreeNodeData>* get_root=tree->get_root();
	get_root->set_children(insert_children);
	get_root->add_child(child3);

	CDynamicObjectArray* get_children=get_root->get_children();
	EXPECT_EQ(get_children->get_num_elements(),3);
	CTreeMachineNode<id3TreeNodeData>* get_child2=((CTreeMachineNode<id3TreeNodeData>*)
							 get_children->get_element(1));
	SG_UNREF(get_children);
	get_children=get_child2->get_children();

	EXPECT_EQ(get_child2->data.attribute_id,2);
	EXPECT_EQ(get_child2->data.transit_if_feature_value,2.0);
	EXPECT_EQ(get_child2->data.class_label,22.0);
	EXPECT_EQ(get_child2->machine(),2);
	EXPECT_EQ(get_child2->parent()->machine(),-1);
	EXPECT_EQ(get_children->get_num_elements(),0);

	SG_UNREF(get_child2);
	SG_UNREF(get_root);
	SG_UNREF(tree);
	SG_UNREF(insert_children);
	SG_UNREF(get_children);
}
