/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Delu Zhu
 * Copyright (C) 2012 Delu Zhu
 */

#include <shogun/machine/Machine.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/DotFeatures.h>



namespace shogun
{


/** @brief Class AttributeNode, used in CC45 classifier
 *
 * AttributeNode represents an attribute of the sample.
 * All attribute nodes comprise an attrib list.
 *
 * \f[
 * P(c) \prod_{i} P(x_i|c)
 * \f]
 *
 */
class CAttribNode
{
public:

	/// index in the attrib list
    int32_t id;
    
	/// is discrete attrib or not
	int32_t is_discrete;

	/// values of discrete-valued attrib
    int32_t *value;

	/// size of value array
    int32_t size;

	/// next node in attrib list
    CAttribNode *next;
};



/** @brief Class TreeNode, the node of decision tree.
 *
 * AttributeNode represents an attribute of the sample.
 * All attribute nodes comprise an attrib list.
 *
 * \f[
 * P(c) \prod_{i} P(x_i|c)
 * \f]
 *
 */
class CTreeNode
{
public:
    CTreeNode(int32_t num);
public:

    ///the brother list 
    CTreeNode *list;

	///left child
    CTreeNode *leftNode;

	///right child
    CTreeNode *rightNode;

	///the splitting attribute
    int32_t attrib_id;

	///the splitting threshold if splitting attribute is continuous-valued
    int32_t threshold;  

	///store the index of samples included in this tree node
    int32_t *index; 

	///the size of index array
    int32_t samples_count;   

	///1 represents leaf node, 0 represents other nodes
    int32_t is_leaf;   

	///the major class label of the samples included in this tree node
    int32_t major_class;  

	///the count of samples with minor class label
    int32_t minor_count;  
};



class CDotFeatures;
class CMachine;


/** @brief Class C45, the C4.5 classifier
 *
 *
 * \f[
 * P(c) \prod_{i} P(x_i|c)
 * \f]
 *
 */
class CC45 : public CMachine
{
public:
    CC45():CMachine(), m_features(NULL) {};

    CC45(CFeatures *t_examples, CLabels *t_labels);

    ~CC45();

	/** set features for classify
	 * @param features features to be set
	 */
    inline void set_features(CFeatures *features)
    {
        SG_UNREF(m_features);
        SG_REF(features);
        m_features = (CDotFeatures*)features;
    }

	/** get features for classify
	 * @return current features
	 */
    inline CFeatures *get_features()
    {
        SG_REF(m_features);
        return m_features;
    }

	/** get name
	 * @return classifier name
	 */
    inline const char *get_name() const
    {
        return "C45";
    };

	/** set total information
	 * @param information information to be set
	 */
	void  set_info(float64_t information);

	/** get information
	 * @return total information 
	 */
	float64_t get_info();

	/** build attribute list
	 * @param values values used to build attribute list
	 */
    void set_attribute_list(SGMatrix<int32_t> values);

	/** training samples
	 * @param train_examples samples used to generate a decision tree
	 */
    bool train_machine(CFeatures *train_examples = NULL);

	/** get labels of testing samples
	 * @return labels of testing samples
	 */
    CLabels *apply();

	/** get labels of data
	 * @param data subset of testing samples
	 * @return labels of data
	 */
    CLabels *apply(CFeatures *data);

	/** classifiy specified example
	 * @param idx example index
	 * @return label
	 */
    float64_t apply(int32_t idx);

protected:
	/** generate decision tree with root n
	 * @param n root of this decision tree
	 * @return root of this decision tree
	 */
    CTreeNode *generate_decision_tree(CTreeNode *n);

	/** compute the gain ratio when splitting samples by a
	 * @param a discrete-valued attribute
	 * @return gain ratio of this splitting
	 */
    float64_t compute_gain_ratio_discreteized(CAttribNode *a);

	/** compute the gain ratio when splitting samples by a
	 * @param a continuous-valued attribute
	 * @param max_split splitting point with max gain ratio
	 * @return max gain ratio of all possible splitting on a
	 */
    float64_t compute_gain_ratio_continuous(CAttribNode *a, int32_t &max_split);

	/** get attribute list
	 * @return head node of attribute list
	 */
    CAttribNode *get_attribute_list();

	/** depth first search: get label of specified example
	 * @param r root of subtree
	 * @param k example index
	 * @return label of example k
	 */
    float64_t dfs(CTreeNode *r,int k);


private:

	/** sort values of the given continuous attribute
	 * @param idx attrib index
	 * @param left left boundary of the array
	 * @param right right boundary of the array
	 */
	void quicksort(int32_t idx, int32_t left, int32_t right);

	/** helper function used in quicksort 
	 * @param idx attrib index
	 * @param left left boundary of the array
	 * @param right right boundary of the array
	 * @return partition idx
	 */
    int32_t partition(int32_t idx, int32_t left, int32_t right);

protected:

	/// number of training samples
    int32_t m_num_samples;

	/// dimension of feature/attribute
    int32_t m_dim;

	/// amount of information
    float64_t info;

	/// root of decision tree
    CTreeNode *root;

	/// current tree node
    CTreeNode *current_tree_node;

	/// features for training or classifying
    CDotFeatures *m_features;

	/// feature matrix for training or classifying
    SGMatrix<float64_t> feature_matrix;

	/// used in compute_gain_ratio_continuous, stores each value and responding label of continuous attribute
    SGMatrix<float64_t> *value_label;

	/// labels of features
    SGVector<int32_t> train_labels;

	/// list of attributes
    CAttribNode *attribute_list;

	/// vector filled with 0/1, 1 represents specified element is discrete-valued  
    SGVector<int32_t> *is_discrete;

	///used in compute_gain_ratio_continuous
	///yes_number[0],yes_number[1] stores number of samples with one label included in left child and right child respectively
    SGVector<int32_t> *yes_num;

	///used in compute_gain_ratio_continuous
	///yes_number[1],no_number[1] stores number of samples with the other label included in left child and right child respectively
    SGVector<int32_t> *no_num;
};

}
