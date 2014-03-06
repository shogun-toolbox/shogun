/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2013 Monica Dragan
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

#include <shogun/mathematics/Math.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/tree/ID3ClassifierTree.h>
#include <vector>

using namespace std;
using namespace shogun;

CID3ClassifierTree::CID3ClassifierTree()
: CGenericTreeMachine<id3TreeNodeData>()
{
}

CID3ClassifierTree::~CID3ClassifierTree()
{
}

float64_t CID3ClassifierTree::informational_gain_attribute(int32_t attr_no, CFeatures* data, 
								CMulticlassLabels* class_labels)
{
	REQUIRE(data,"data required for information gain calculation")
	REQUIRE(data->get_feature_class()==C_DENSE,
		"Dense data required for information gain calculation")

	float64_t gain = 0;
	CDenseFeatures<float64_t>* feats = (CDenseFeatures<float64_t>*) data;
	int32_t num_vecs = feats->get_num_vectors();

	//get column of attribute attr_no
	SGVector<float64_t> attribute_values = feats->get_transposed()->get_feature_vector(attr_no);
	CMulticlassLabels* attribute_labels = new CMulticlassLabels(attribute_values);

	//for each attribute label
	for(int32_t i=0; i<attribute_labels->get_unique_labels().size(); i++)	
	{
		//calculate the number of labels with the specific label
		//iterate through all attribute labels	
		int32_t attr_cnt = 0;	
		for(int32_t j=0; j<num_vecs; j++)
		{
			if(attribute_labels->get_unique_labels()[i] == attribute_labels->get_label(j))
				attr_cnt++;
		}
	
		float64_t label_entropy = 0;		
		//calculate class entropy for the specific label of the attribute
		//for each class
		for(int32_t j=0; j<class_labels->get_unique_labels().size(); j++)
		{	
			//iterate through all class labels of the dataset
			int32_t class_cnt = 0;
			for(int32_t k=0; k<num_vecs; k++)
			{
				if(attribute_labels->get_unique_labels()[i]
						 == attribute_labels->get_label(k))
				{
					if(class_labels->get_unique_labels().get_element(j)
								 == class_labels->get_label(k))
						class_cnt++;				
				}
			}
			float64_t ratio = (float64_t)class_cnt/attr_cnt;
			
			if(ratio != 0)				
				label_entropy -= ((float64_t)ratio)*(CMath::log2(ratio));			
		}

		gain += (float64_t)((float64_t)attr_cnt/attribute_labels->
							get_num_labels())*label_entropy;
	
	}

	float64_t data_entropy = entropy(class_labels);
	gain = data_entropy- gain;
	
	return gain;	
}

float64_t CID3ClassifierTree::entropy(CMulticlassLabels *labels)
{
	float64_t entr = 0;

	//for each class
	for(int32_t i=0;i<labels->get_unique_labels().size();i++)
	{	
		//iterate through all the labels of the dataset
		//compute the number of features that belong to the specific class
		int32_t cnt = 0;		
		for(int32_t j=0;j<labels->get_num_labels();j++)
		{				
			if(labels->get_unique_labels().get_element(i) == labels->get_label(j))
				cnt++;			
		}
		float64_t ratio = (float64_t)cnt/labels->get_num_labels();
				
		if(ratio != 0)				
			entr -= ((float64_t)ratio)*(CMath::log2(ratio));			
	}
	
	return entr;
	
}

bool CID3ClassifierTree::train_machine(CFeatures* data)
{
	REQUIRE(data,"data required for training")
	REQUIRE(data->get_feature_class()==C_DENSE, "Dense data required for training")

	int32_t num_features = ((CDenseFeatures<float64_t>*) data)->get_num_features();
	SGVector<int32_t> feature_ids = SGVector<int32_t>(num_features);
	for (int32_t i=0; i<num_features; i++)
		feature_ids[i] = i;

	m_root = id3train(data, (CMulticlassLabels*) m_labels, feature_ids, 0);

	return true;
}

CGenericTreeMachineNode<id3TreeNodeData>* CID3ClassifierTree::id3train(CFeatures* data, 
	CMulticlassLabels* class_labels, SGVector<int32_t> feature_id_vector, int32_t level)
{	
	CGenericTreeMachineNode<id3TreeNodeData>* node = 
			new CGenericTreeMachineNode<id3TreeNodeData>();

	CDenseFeatures<float64_t>* feats = (CDenseFeatures<float64_t>*) data;

	//if all samples belong to the same class
	if(class_labels->get_unique_labels().size() == 1)
	{
		node->data.class_label=class_labels->get_unique_labels()[0];
		return node;
	}

	//if only one feature is left
	if(feature_id_vector.vlen == 0)
	{
		/* Not Implemented yet */
		return node;
	}

	
	//else get the feature with the highest informational gain
	float64_t max = 0;
	int32_t best_feature_index = -1;
	for(int32_t i=0; i<feats->get_num_features(); i++)
	{
		float64_t gain = informational_gain_attribute(i,feats,class_labels);	

		if(gain > max){
			max = gain;
			best_feature_index = i;
		}
	}	
	
	
	//get feature values for the best feature chosen
	SGVector<float64_t> best_feature_values =  feats->get_transposed()
					->get_feature_vector(best_feature_index);
	CMulticlassLabels* best_feature_labels = new CMulticlassLabels(best_feature_values);
	SGVector<float64_t> best_labels_unique = best_feature_labels->get_unique_labels();

	for(int32_t i=0; i<best_labels_unique.vlen; i++)
	{
		//compute the number of attributes with the current attribute values
		//to allocate matrix
		int32_t num_cols = 0;
		float64_t active_feature_value = best_labels_unique[i];

		for(int32_t j=0; j<feats->get_num_vectors(); j++)
		{
			if( active_feature_value == best_feature_values[j])
			{
				num_cols++;
			}
		}

		SGMatrix<float64_t> mat = SGMatrix<float64_t>(feats->get_num_features()-1,
										 num_cols);	
		SGVector<float64_t> new_labels_vector = SGVector<float64_t>(num_cols);

		int32_t cnt = 0;
		//choose the samples that have the specific value for the attribute best_attribute
		for(int32_t j=0; j<feats->get_num_vectors(); j++)
		{
			SGVector<float64_t> sample = feats->get_feature_vector(j);
			if(active_feature_value == sample[best_feature_index])
			{
				int32_t idx = -1;
				for(int32_t k=0; k<sample.size(); k++)
				{
					if(k != best_feature_index)			
						mat(++idx, cnt) = sample[k];
				}

				new_labels_vector[cnt] = class_labels->get_labels()[j];
				cnt++;					
			}
		}

		CMulticlassLabels* new_class_labels = new CMulticlassLabels(new_labels_vector);

		//remove the best_attribute from the remaining attributes index vector
		SGVector<int32_t> new_feature_id_vector = SGVector<int32_t>(feature_id_vector.vlen-1);		
		cnt = -1;
		for(int32_t j=0;j<feature_id_vector.vlen;j++)
		{
			if(j!=best_feature_index)
				new_feature_id_vector[++cnt] = feature_id_vector[j];		
		}

		CDenseFeatures<float64_t>* new_data = new CDenseFeatures<float64_t>(mat);

		CGenericTreeMachineNode<id3TreeNodeData>* child = id3train(new_data, new_class_labels,
									 new_feature_id_vector, level+1);
		child->data.transitIf_feature_value = active_feature_value;
		child->set_parent(node);
		node->data.attribute_id = feature_id_vector[best_feature_index];
		node->add_child(child);

	}		

	return node;
}

CMulticlassLabels* CID3ClassifierTree::apply_multiclass(CFeatures* data)
{
	REQUIRE(data, "Data required for classification in apply_multiclass")

	CDenseFeatures<float64_t>* feats = (CDenseFeatures<float64_t>*) data;
	int32_t num_vecs = feats->get_num_vectors();
	SGVector<float64_t> labels = SGVector<float64_t>(num_vecs);

	for (int32_t i=0; i<num_vecs; i++)
	{
		SGVector<float64_t> sample = feats->get_feature_vector(i);
		CGenericTreeMachineNode<id3TreeNodeData>* node = m_root;
		vector<CGenericTreeMachineNode<id3TreeNodeData>*> children 
							= node->get_children();

		while (children.size())
		{
			int32_t flag = 0;
			for (int32_t j=0; j<children.size(); j++)
			{
				if (children[j]->data.transitIf_feature_value == 
							sample[node->data.attribute_id])
				{
					flag = 1;
					node = children[j];
					children = node->get_children();
					break;
				}
			}

			if (!flag)
				break;
		}
		
		labels[i] = node->data.class_label;
	}
	
	CMulticlassLabels* ret = new CMulticlassLabels(labels);
	return ret;
}
