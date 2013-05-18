#include <limits>
#include <queue>
#include <algorithm>
#include <functional>
#include <stdlib.h>
#include <string>

#include <shogun/labels/BinaryLabels.h>
#include <shogun/multiclass/tree/RelaxedTreeUtil.h>
#include <shogun/multiclass/tree/RelaxedTree.h>
#include <shogun/kernel/GaussianKernel.h>


#include <shogun/labels/MulticlassLabels.h>
#include <shogun/io/streaming/StreamingAsciiFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DenseSubsetFeatures.h>
#include <shogun/base/init.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/transfer/multitask/MultitaskKernelTreeNormalizer.h>


#include "classifier_multiclass_id3.h"


using namespace shogun;

CID3Classifier::CID3Classifier()
	:m_feats(NULL), m_num_classes(0)
{

}

CID3Classifier::~CID3Classifier()
{

}


float64_t CID3Classifier::informational_gain_attribute(int32_t attr_no, CFeatures* data, CMulticlassLabels *class_labels)
{
	int32_t i, j, k;
	float64_t gain = 0;
	CDenseFeatures<float64_t> *feats;	

	if (data)
	{
		feats = dynamic_cast<CDenseFeatures<float64_t>*>(data);
		if (feats == NULL){
			SG_SPRINT("Require non-NULL dense features of float64_t\n");
			return 0;
		}
	}
	
	//get column of attribute attr_no
	SGVector<float64_t> attribute_values =  feats->get_transposed()->get_feature_vector(attr_no);
	CMulticlassLabels* attribute_labels = new CMulticlassLabels(attribute_values);
	
	//for each attribute label
	for(i=0; i<attribute_labels->get_unique_labels().size(); i++)	
	{
		//calculate the number of labels with the specific label
		//iterate through all attribute labels	
		int32_t attr_cnt = 0;	
		for(j=0; j<attribute_labels->get_num_labels(); j++)
		{
			if(attribute_labels->get_unique_labels().get_element(i) == attribute_labels->get_label(j))
				attr_cnt++;
		}		
	
		float64_t label_entropy = 0;		
		//calculate class entropy for the specific label of the attribute
		//for each class
		for(j=0; j<class_labels->get_unique_labels().size(); j++)
		{	
			//iterate through all class labels of the dataset
			//compute the number of features with lable i for attribute attr_no that belong to the specific class
			int32_t class_cnt = 0;	

			for(k=0; k<class_labels->get_num_labels(); k++)
			{				
				if(attribute_labels->get_unique_labels().get_element(i) == attribute_labels->get_label(k))
					if(class_labels->get_unique_labels().get_element(j) == class_labels->get_label(k))
						class_cnt++;				
			}
			float64_t ratio = (float64_t)class_cnt/attr_cnt;
			
			if(ratio != 0)				
				label_entropy -= ((float64_t)ratio)*((float64_t)log10(ratio)/log10(2));			
		}

		gain += (float64_t)((float64_t)attr_cnt/attribute_labels->get_num_labels())*label_entropy;
	
	}

	float64_t data_entropy = entropy(class_labels);
	gain = data_entropy- gain;
	
	return gain;
	
}

float64_t CID3Classifier::entropy(CMulticlassLabels *labels)
{
	int32_t i, j, k;
	float64_t entropy = 0;

	//for each class
	for(i=0;i<labels->get_unique_labels().size();i++)
	{	
		//iterate through all the labels of the dataset
		//compute the number of features that belong to the specific class
		int32_t cnt = 0;		
		for(j=0;j<labels->get_num_labels();j++)
		{				
			if(labels->get_unique_labels().get_element(i) == labels->get_label(j))
				cnt++;			
		}
		float64_t ratio = (float64_t)cnt/labels->get_num_labels();
				
		if(ratio != 0)				
			entropy -= ((float64_t)ratio)*((float64_t)log10(ratio)/log10(2));	
		
	}
	
	return entropy;
	
}

id3_node CID3Classifier::train(CFeatures* data, CMulticlassLabels *class_labels, SGVector<int32_t> *attributes, int level)
{

	int i,j,k;
	float max = 0;
	int best_attribute = 0;
	
	id3_node node;
	node.attribute = -1;	
	node.attribute_label = -1;
	node.parent = NULL;

	CDenseFeatures<float64_t> *feats;
		
	if (data)
	{
		feats = dynamic_cast<CDenseFeatures<float64_t>*>(data);
		if (feats == NULL){
			SG_SPRINT("Require non-NULL dense features of float64_t\n");
			return node;
		}
	}
	
	if(attributes == NULL){
	
		attributes = new SGVector<int32_t>(feats->get_num_features());					
		for(i=0;i<attributes->size();i++)
			attributes->set_element(i,i);	
	
	}
	
	//if all samples belong to the same class
	if(class_labels->get_unique_labels().size() == 1){

		node.attribute=class_labels->get_unique_labels().get_element(0);
		return node;
	}
	
	//if training set is vide
	else if(class_labels->get_unique_labels().size() == 0){

		return node;	
	}
	
	//if there is no attribute left
	else if(feats->get_num_features() == 0){

		return node;		
	}
	
	//else get the attribute with the highest informational gain
	for(i=0; i<feats->get_num_features(); i++){
		float64_t gain = informational_gain_attribute(i,feats,class_labels);	

		if(gain > max){
			max = gain;
			best_attribute = i;
		}
	}	
	
	
	//get column of attribute attr_no
	SGVector<float64_t> attribute_values =  feats->get_transposed()->get_feature_vector(best_attribute);
	CMulticlassLabels* attribute_labels = new CMulticlassLabels(attribute_values);	
	
	for(i=0; i<attribute_labels->get_unique_labels().size(); i++)
	{
	
		//comupte the number attributes with the curret attribute values
		//to allocate matrix
		int32_t no_lines = 0;
		for(j=0; j<feats->get_num_vectors(); j++){
			if(attribute_labels->get_unique_labels().get_element(i) == feats->get_feature_vector(j).get_element(best_attribute))
			{
				no_lines++;
			}
		}

		SGMatrix<float64_t> mat = SGMatrix<float64_t>(feats->get_num_features()-1, no_lines);	
		CMulticlassLabels* new_class_labels = new CMulticlassLabels(no_lines);	
								
		int32_t cnt = -1;
		
		//choose the samples that have the specific value for the attribute best_attribute
		for(j=0; j<feats->get_num_vectors(); j++){
			SGVector< float64_t > sample = feats->get_feature_vector(j);
			if(attribute_labels->get_unique_labels().get_element(i) == sample.get_element(best_attribute))
			{
				SGVector<float64_t> sample2(sample.size()-1);

				int idx = -1;
				for(k=0; k<sample.size(); k++)
				{
					if(k != best_attribute){						
						sample2.set_element(sample.get_element(k),++idx);
					}
				}

				std::copy(sample2.vector, sample2.vector+sample2.vlen, mat.get_column_vector(++cnt));
				new_class_labels->set_label(cnt,class_labels->get_label(j));	
									
			}
		}
		
		//remove the best_attribute from the remaining attributes index vector
		SGVector<int32_t>* new_attributes = new SGVector<int32_t>(attributes->size()-1);		
		cnt = -1;
		for(j=0;j<attributes->size();j++)
			if(j!=best_attribute)
				new_attributes->set_element(attributes->get_element(j),++cnt);		
		
		CDenseFeatures< float64_t >* new_data = new CDenseFeatures<float64_t>(mat);
		
		id3_node child= train(new_data, new_class_labels, new_attributes, level+1);
		child.attribute_label = i;
		child.parent = (id3_node*)malloc(sizeof(id3_node));
		child.parent->attribute = attributes->get_element(best_attribute);
		node.attribute = attributes->get_element(best_attribute);		
		
		node.children.push_back(child);

	}		

	return node;
}

int CID3Classifier:: evaluate(SGVector<float64_t> sample, id3_node node)
{
	int i;
			
	if(node.children.size() == 0)
	{
			return node.attribute;
	}			
				
	float64_t value = sample.get_element(node.attribute);

	for(i=0; i<node.children.size(); i++)
	{
		if(node.children.at(i).attribute_label == value)
		{
			return evaluate(sample, node.children.at(i));
		}
	}

}

void CID3Classifier::print_dataset(CFeatures* data, CMulticlassLabels *class_labels)
{
	int i, j;
	CDenseFeatures<float64_t> *feats = dynamic_cast<CDenseFeatures<float64_t>*>(data);
	
	SG_SPRINT("\n----------------Dataset print------------------\n");	
	
	for(i=0; i<feats->get_num_vectors(); i++){
	
		SGVector<float64_t> sample = feats->get_feature_vector(i);
		
		for(j=0; j<sample.size(); j++)
			SG_SPRINT("%.0f ",sample.get_element(j));
		SG_SPRINT(" | %.0f ",class_labels->get_label(i));
		SG_SPRINT("\n");
	}
	
	SG_SPRINT("------------------------------------------------\n\n");	
}

void CID3Classifier::print_id3_tree(id3_node* root, int tab)
{
	id3_node* node = root;
	int i;
	
	if(tab == 0)
		SG_SPRINT("\n-----------------ID3 tree print-----------------\n");
	
	if(node != NULL){
		for(i=0;i<tab;i++)
			SG_SPRINT("\t\t");
			
		if(node->parent != NULL){
			SG_SPRINT("-- %s -- ", (names.at(node->parent->attribute)).at(node->attribute_label).c_str());
		}
		
		if(node->children.size() == 0)
		{
				SG_SPRINT("%d\n", node->attribute);
		}
		else{
			SG_SPRINT("%s\n",attribute_names.at(node->attribute).c_str());
		
			for(i=0; i<node->children.size(); i++)
			{
				print_id3_tree(&node->children.at(i), tab+1);
			}
		}
	}
	
	if(tab == 0)	
		SG_SPRINT("------------------------------------------------\n\n");	
}

int main(){

	init_shogun_with_defaults();
	
	int32_t num_vectors = 0;
	int32_t num_feats   = 0;
	
	const char*fname_train = "data/classifier_id3_meteo.dat";

	/** lable names **/
	std::vector<string> attr_names;
	std::vector< std::vector<string> > label_names;	

	attr_names.push_back("outlook");
    std::vector<std::string> attr1;
    attr1.push_back("sunny");
    attr1.push_back("overcast");
    attr1.push_back("rain");        
    	
	attr_names.push_back("temperature");
    std::vector<std::string> attr2;
    attr2.push_back("worm");     
    attr2.push_back("fine");         
    attr2.push_back("cold");         
    	
	attr_names.push_back("humidity");	
    std::vector<std::string> attr3;
    attr3.push_back("high");         
    attr3.push_back("normal");         
    	
	attr_names.push_back("wind");	
    std::vector<std::string> attr4;
    attr4.push_back("true");
    attr4.push_back("false");    

	label_names.push_back(attr1);
	label_names.push_back(attr2);
	label_names.push_back(attr3);
	label_names.push_back(attr4);			
	
	CStreamingAsciiFile *train_file = new CStreamingAsciiFile(fname_train);
	SG_REF(train_file);

	CStreamingDenseFeatures<float64_t> *stream_features = new CStreamingDenseFeatures<float64_t>(train_file, true, 1024);
	SG_REF(stream_features);

	SGMatrix<float64_t> mat;
	SGVector<float64_t> labvec(1000);

	stream_features->start_parser();
	SGVector< float64_t > vec;
	int32_t num_vec=0;
	
	while (stream_features->get_next_example())
	{
		vec = stream_features->get_vector();
		if (num_feats == 0)
		{
			num_feats = vec.vlen;
			mat = SGMatrix<float64_t>(num_feats, 1000);
		}
		std::copy(vec.vector, vec.vector+vec.vlen, mat.get_column_vector(num_vectors));
		labvec[num_vectors] = stream_features->get_label();
		num_vectors++;
		stream_features->release_example();
		num_vec++;

		if (num_vec > 20000)
			break;
	}
	
	stream_features->end_parser();
	mat.num_cols = num_vectors;
	labvec.vlen = num_vectors;
	
	CMulticlassLabels* labels = new CMulticlassLabels(labvec);
	SG_REF(labels);

	// Create features with the useful values from mat
	CDenseFeatures< float64_t >* features = new CDenseFeatures<float64_t>(mat);
	SG_REF(features);

	CID3Classifier *machine = new CID3Classifier();
	machine->set_labels(labels);
	machine->set_features(features);
	machine->set_names(attr_names, label_names);

	id3_node root = machine->train(features, labels);
	machine->print_id3_tree(&root);
	
	//Test the decission tree 
	SGVector<float64_t> input(4);
	input.set_element(2,0);
	input.set_element(2,1);
	input.set_element(1,2);
	input.set_element(1,3);			

	SG_SPRINT("Classification answer: %d\n",machine->evaluate(input, root));
	
	SGVector<float64_t> input2(4);
	input2.set_element(2,0);
	input2.set_element(1,1);
	input2.set_element(0,2);
	input2.set_element(0,3);			

	SG_SPRINT("Classification answer: %d\n",machine->evaluate(input2, root));	
	


	return 0;

}


