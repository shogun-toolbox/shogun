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

#include <shogun/mathematics/Math.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/tree/ID3ClassifierTree.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

CID3ClassifierTree::CID3ClassifierTree()
: CTreeMachine<id3TreeNodeData>()
{
}

CID3ClassifierTree::~CID3ClassifierTree()
{
}

CMulticlassLabels* CID3ClassifierTree::apply_multiclass(CFeatures* data)
{
	REQUIRE(data, "Data required for classification in apply_multiclass\n")

	CDenseFeatures<float64_t>* feats = dynamic_cast<CDenseFeatures<float64_t>*>(data);
	int32_t num_vecs = feats->get_num_vectors();
	SGVector<float64_t> labels = SGVector<float64_t>(num_vecs);

	for (int32_t i=0; i<num_vecs; i++)
	{
		SGVector<float64_t> sample = feats->get_feature_vector(i);
		node_t* node = get_root();
		CDynamicObjectArray* children = node->get_children();

		while (children->get_num_elements())
		{
			int32_t flag = 0;
			for (int32_t j=0; j<children->get_num_elements(); j++)
			{
				node_t* child = dynamic_cast<node_t*>(children->get_element(j));
				if (child->data.transit_if_feature_value 
						== sample[node->data.attribute_id])
				{
					flag = 1;

					SG_UNREF(node);
					node = child;

					SG_UNREF(children);
					children = node->get_children();

					break;
				}

				SG_UNREF(child);
			}

			if (!flag)
				break;
		}
		
		labels[i] = node->data.class_label;

		SG_UNREF(node);
		SG_UNREF(children);
	}
	
	CMulticlassLabels* ret = new CMulticlassLabels(labels);
	return ret;
}

bool CID3ClassifierTree::train_machine(CFeatures* data)
{
	REQUIRE(data,"Data required for training\n")
	REQUIRE(data->get_feature_class()==C_DENSE, "Dense data required for training\n")

	int32_t num_features = (dynamic_cast<CDenseFeatures<float64_t>*>(data))->get_num_features();
	SGVector<int32_t> feature_ids = SGVector<int32_t>(num_features);
	feature_ids.range_fill();

	set_root(id3train(data, dynamic_cast<CMulticlassLabels*>(m_labels), feature_ids, 0));

	return true;
}

CTreeMachineNode<id3TreeNodeData>* CID3ClassifierTree::id3train(CFeatures* data, 
	CMulticlassLabels* class_labels, SGVector<int32_t> feature_id_vector, int32_t level)
{
	node_t* node = new node_t();
	CDenseFeatures<float64_t>* feats = dynamic_cast<CDenseFeatures<float64_t>*>(data);
	int32_t num_vecs = feats->get_num_vectors();

	// if all samples belong to the same class
	if (class_labels->get_unique_labels().size() == 1)
	{
		node->data.class_label = class_labels->get_unique_labels()[0];
		return node;
	}

	// if no feature is left
	if (feature_id_vector.vlen == 0)
	{
		// decide label - label occuring max times
		SGVector<float64_t> labels = class_labels->get_labels();
		labels.qsort();

		int32_t most_label = labels[0];
		int32_t most_num = 1;
		int32_t count = 1;

		for (int32_t i=1; i<labels.vlen; i++)
		{
			while ((labels[i] == labels[i-1]) && (i<labels.vlen))
			{
				count++;
				i++;
			}

			if (count>most_num)
			{
				most_num = count;
				most_label = labels[i-1];
			}

			count = 1;
		}

		node->data.class_label = most_label;
		return node;
	}

	// else get the feature with the highest informational gain
	float64_t max = 0;
	int32_t best_feature_index = -1;
	for (int32_t i=0; i<feats->get_num_features(); i++)
	{
		float64_t gain = informational_gain_attribute(i,feats,class_labels);

		if (gain >= max)
		{
			max = gain;
			best_feature_index = i;
		}
	}	

	//get feature values for the best feature chosen
	SGVector<float64_t> best_feature_values = SGVector<float64_t>(num_vecs);
	for (int32_t i=0; i<num_vecs; i++)
		best_feature_values[i] = (feats->get_feature_vector(i))[best_feature_index];

	CMulticlassLabels* best_feature_labels = new CMulticlassLabels(best_feature_values);
	SGVector<float64_t> best_labels_unique = best_feature_labels->get_unique_labels();

	for (int32_t i=0; i<best_labels_unique.vlen; i++)
	{
		//compute the number of vectors with active attribute value
		int32_t num_cols = 0;
		float64_t active_feature_value = best_labels_unique[i];

		for (int32_t j=0; j<num_vecs; j++)
		{
			if ( active_feature_value == best_feature_values[j])
				num_cols++;
		}

		SGMatrix<float64_t> mat = SGMatrix<float64_t>(feats->get_num_features()-1, num_cols);	
		SGVector<float64_t> new_labels_vector = SGVector<float64_t>(num_cols);

		int32_t cnt = 0;
		//choose the samples that have the active feature value
		for (int32_t j=0; j<num_vecs; j++)
		{
			SGVector<float64_t> sample = feats->get_feature_vector(j);
			if (active_feature_value == sample[best_feature_index])
			{
				int32_t idx = -1;
				for (int32_t k=0; k<sample.size(); k++)
				{
					if (k != best_feature_index)			
						mat(++idx, cnt) = sample[k];
				}

				new_labels_vector[cnt] = class_labels->get_labels()[j];
				cnt++;					
			}
		}

		//remove the best_attribute from the remaining attributes index vector
		SGVector<int32_t> new_feature_id_vector = SGVector<int32_t>(feature_id_vector.vlen-1);		
		cnt = -1;
		for (int32_t j=0;j<feature_id_vector.vlen;j++)
		{
			if (j!=best_feature_index)
				new_feature_id_vector[++cnt] = feature_id_vector[j];		
		}

		CMulticlassLabels* new_class_labels = new CMulticlassLabels(new_labels_vector);
		CDenseFeatures<float64_t>* new_data = new CDenseFeatures<float64_t>(mat);

		node_t* child = id3train(new_data, new_class_labels, new_feature_id_vector, level+1);
		child->data.transit_if_feature_value = active_feature_value;
		node->data.attribute_id = feature_id_vector[best_feature_index];
		node->add_child(child);

		SG_UNREF(new_class_labels);
		SG_UNREF(new_data);
	}

	SG_UNREF(best_feature_labels);

	return node;
}

float64_t CID3ClassifierTree::informational_gain_attribute(int32_t attr_no, CFeatures* data, 
								CMulticlassLabels* class_labels)
{
	REQUIRE(data,"Data required for information gain calculation\n")
	REQUIRE(data->get_feature_class()==C_DENSE,
		"Dense data required for information gain calculation\n")

	float64_t gain = 0;
	CDenseFeatures<float64_t>* feats = dynamic_cast<CDenseFeatures<float64_t>*>(data);
	int32_t num_vecs = feats->get_num_vectors();

	//get attribute values for attribute
	SGVector<float64_t> attribute_values = SGVector<float64_t>(num_vecs);

	for (int32_t i=0; i<num_vecs; i++)
		attribute_values[i] = (feats->get_feature_vector(i))[attr_no];

	CMulticlassLabels* attribute_labels = new CMulticlassLabels(attribute_values);
	SGVector<float64_t> attr_val_unique = attribute_labels->get_unique_labels();

	for (int32_t i=0; i<attr_val_unique.vlen; i++)
	{
		//calculate class entropy for the specific attribute_value
		int32_t attr_count=0;

		for (int32_t j=0; j<num_vecs; j++)
		{
			if (attribute_values[j] == attr_val_unique[i])
				attr_count++;
		}

		SGVector<float64_t> sub_class = SGVector<float64_t>(attr_count);
		int32_t count = 0;

		for (int32_t j=0; j<num_vecs; j++)
		{
			if (attribute_values[j] == attr_val_unique[i])
				sub_class[count++] = class_labels->get_label(j);
		}
		
		CMulticlassLabels* sub_labels = new CMulticlassLabels(sub_class);
		float64_t sub_entropy = entropy(sub_labels);
		gain += sub_entropy*(attr_count-0.f)/(num_vecs-0.f);

		SG_UNREF(sub_labels);
	}

	float64_t data_entropy = entropy(class_labels);
	gain = data_entropy-gain;

	SG_UNREF(attribute_labels);
	
	return gain;
}

float64_t CID3ClassifierTree::entropy(CMulticlassLabels* labels)
{
	SGVector<float64_t> log_ratios = SGVector<float64_t>
			(labels->get_unique_labels().size());

	for (int32_t i=0;i<labels->get_unique_labels().size();i++)
	{
		int32_t count = 0;

		for (int32_t j=0;j<labels->get_num_labels();j++)
		{
			if (labels->get_unique_labels()[i] == labels->get_label(j))
					count++;
		}

		log_ratios[i] = (count-0.f)/(labels->get_num_labels()-0.f);

		if (log_ratios[i] != 0)
			log_ratios[i] = CMath::log(log_ratios[i]);			
	}

	return CStatistics::entropy(log_ratios.vector, log_ratios.vlen);
}
